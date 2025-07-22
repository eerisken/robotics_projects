import py_trees
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

import matplotlib.pyplot as plt

class PickJar(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_helper, yolo_model_path):
        super(PickJar, self).__init__(name)
        self.name = name
        self.robot = robot_helper
        
        self.arm_joints = robot_helper.arm_joints
        self.arm_chain = robot_helper.arm_chain
        
        self.joint_names = robot_helper.arm_joint_names
        self.gripper_motors = robot_helper.gripper_motors

        self.timestep = robot_helper.timestep
        self.stage = "detect"
        self.target_angles = None
        
        self.initialised = False
        
    def initialise(self):
        if not self.initialised:
            self.camera = self.robot.getDevice('camera')
            self.camera.enable(32)
            self.camera.recognitionEnable(32)
            
            self.camera_node = self.robot.getFromDef("CAMERA")
            self.arm_front_node = self.robot.getFromDef("ARM_FRONT_EXTENSION")
            
            self.left_motor = self.robot.left_motor
            self.right_motor = self.robot.right_motor
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
    
            self.gripper_left = self.robot.getDevice('gripper_left_finger_joint')
            self.gripper_right = self.robot.getDevice('gripper_right_finger_joint')
            self.gripper_left.enableForceFeedback(32)
            self.gripper_right.enableForceFeedback(32)
            self.gripper_left.setVelocity(0.03)
            self.gripper_right.setVelocity(0.03)
    
            self.torso_joint = self.robot.torso_joint
            self.torso_sensor = self.torso_joint.getPositionSensor()
            self.torso_sensor.enable(32)
    
            self.arm_4_joint = self.robot.getDevice('arm_4_joint')
            self.arm_4_sensor = self.arm_4_joint.getPositionSensor()
            self.arm_4_sensor.enable(32)
            
            self.torso_joint.setVelocity(0.05)
            self.arm_4_joint.setVelocity(0.5)
            self.initialised = True


    def update(self):
        if self.stage == "detect":
            recognized_objects = self.camera.getRecognitionObjects()
            for obj in recognized_objects:
                self.logger.debug("Recognized object:")
                self.logger.debug(f"  Model: {obj.getModel()}")
                self.logger.debug(f"  Position (camera frame): {obj.getPosition()}")
                self.logger.debug(f"  Size: {obj.getSize()}")
                self.logger.debug(f"  Colors: {obj.getColors()}")
            
            # Get the object's position in camera coordinates
            #pose_cam = np.array(recognized_objects[0].getPose()).reshape(4, 4)  # camera-relative
            # Position in camera frame
            pos = recognized_objects[0].getPosition()  # [x, y, z]
            # Orientation in camera frame (3x3)
            quat = np.array(recognized_objects[0].getOrientation())      # [qx, qy, qz, qw]
            # Convert quaternion to rotation matrix
            rot = R.from_quat(quat).as_matrix()        # shape (3, 3)
           
            # Build 4x4 pose matrix in camera frame
            pose_cam = np.eye(4)
            pose_cam[:3, :3] = rot
            pose_cam[:3, 3] = pos
            
            camera_world = np.array(self.camera_node.getPose()).reshape(4, 4)
            arm_base_world = np.array(self.arm_front_node.getPose()).reshape(4, 4)
            
            pose_world = camera_world @ pose_cam
            
            arm_base_inv = np.linalg.inv(arm_base_world)
            pose_base = arm_base_inv @ pose_world
            
            target_for_ik = pose_base[:3, 3]  # x, y, z in arm base frame

            ik_angles = self._compute_ik(target_for_ik)
            if ik_angles is None:
                return py_trees.common.Status.FAILURE

            self.target_angles = ik_angles
            self._move_arm(self.target_angles)
            self.stage = "move_arm"
            
            return py_trees.common.Status.RUNNING

        elif self.stage == "move_arm":
            if self._arm_reached_target():
                # Close gripper
                self.gripper_motor.setVelocity(1.0)
                self.gripper_motor.setPosition(0.0)  # Adjust to close position
                self.stage = "grasp"
            return py_trees.common.Status.RUNNING

        elif self.stage == "grasp":
            # Could add delay or sensor feedback here
            self.stage = "done"
            return py_trees.common.Status.SUCCESS

        else:
            return py_trees.common.Status.SUCCESS

    def _get_rgb_image(self):
        image = self.robot.get_camera_image()

        if hasattr(self.robot, 'display') and self.robot.display:
            self.robot.set_display_image(image)

        return image

    def _detect_jars(self, image):
        # YOLO expects RGB image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLO model (ultralytics handles resizing internally)
        results = self.yolo_model(image_rgb, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.yolo_model.names[class_id]
                confidence = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                self.logger.debug(f"Detected: {class_name}, Confidence: {confidence:.2f}, Box: {xyxy}")

        jars = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                label = self.yolo_model.names[class_id]

                if label.lower() in ["jar", "bottle", "cup"]:  # Filter for jars (or retrain model for this)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    jars.append((x_center, y_center, w, h))

        return jars

    def _get_depth_at_pixel(self, x, y):
        if self.camera_depth is None:
            return None
        width = self.camera_depth.getWidth()
        height = self.camera_depth.getHeight()
        if x < 0 or x >= width or y < 0 or y >= height:
            return None
        
        depth_image = self.camera_depth.getRangeImage()
        if not depth_image:
            self.logger.debug(f"{self.name} Depth Image not available {depth_image}") 
        #else:
        #    self.logger.debug(f"{self.name} Depth Image available {len(depth_image)}")    
        
        #self.plot_depth_image(depth_image)
        
        depth = depth_image[y * width + x]
        self.logger.debug(f"{self.name} Depth values {x}, {y}, {width}, {depth}, {len(depth_image)}")    
        
        if np.isnan(depth) or np.isinf(depth) or depth <= 0.0:
            return None
        
        return depth
    
    def plot_depth_image(self, depth_image):
        if self.camera_depth is None or self.robot.display is None:
            return

        width = self.camera_depth.getWidth()
        height = self.camera_depth.getHeight()
        max_range = self.camera_depth.getMaxRange()
    
        # Get and reshape the depth image
        depth = np.array(self.camera_depth.getRangeImage()).reshape((height, width))
    
        # Log inf stats
        inf_count = np.sum(~np.isfinite(depth))
        self.logger.debug(f"{self.name}: Depth image - total: {depth.size}, inf pixels: {inf_count}")
    
        # Replace inf with max range
        depth[~np.isfinite(depth)] = max_range
    
        # Clip and normalize to a displayable range
        min_clip = 0.5
        max_clip = 1.5
        depth_clipped = np.clip(depth, min_clip, max_clip)
        depth_normalized = ((1.0 - (depth_clipped - min_clip) / (max_clip - min_clip)) * 255.0).astype(np.uint8)

    
        # Convert grayscale to RGB
        rgb_image = np.stack([depth_normalized] * 3, axis=-1)
    
        # Convert to byte string for Webots display
        image_bytes = rgb_image.flatten().tobytes()

        handle = self.robot.display.imageNew(image_bytes, self.robot.display.RGB, width, height)
        self.robot.display.imagePaste(handle, 0, 0, False)
      
    def _get_average_depth(self, x, y, window=5):
        valid_depths = []
        for dx in range(-window, window + 1):
            for dy in range(-window, window + 1):
                d = self._get_depth_at_pixel(x + dx, y + dy)
                if d is not None:
                    valid_depths.append(d)
        return np.mean(valid_depths) if valid_depths else None

    def _get_nearest_valid_depth(self, x, y, max_radius=20):
        depth_image = np.array(self.camera_depth.getRangeImage()).reshape(
            (self.camera_depth.getHeight(), self.camera_depth.getWidth())
        )
        h, w = depth_image.shape
    
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        val = depth_image[ny, nx]
                        if np.isfinite(val):
                            self.logger.debug(f"{self.name} Valid depth found at radius {r}, offset ({dx},{dy})")
                            return val
        return None
    
    def _backproject_pixel_to_3d(self, x_pixel, y_pixel, depth):
        #print("_backproject_pixel_to_3d", x_pixel, y_pixel, depth)
        fx = self.camera_intrinsics[0]
        fy = self.camera_intrinsics[1]
        cx = self.camera_intrinsics[2]
        cy = self.camera_intrinsics[3]
        X = (x_pixel - cx) * depth / fx
        Y = (y_pixel - cy) * depth / fy
        Z = depth
        #print("_backproject_pixel_to_3d", [X, Y, Z])
        return np.array([X, Y, Z])

    def _transform_camera_to_robot(self, point_camera):
        t = np.array(self.camera_pose['translation'])
        rot = np.array(self.camera_pose['rotation'])
        
        #q = R.from_matrix(r_matrix).as_quat()
        r = R.from_quat(rot) 
        point_robot = r.apply(point_camera) + t
        return point_robot

    def _compute_ik(self, target_position):
        if np.isnan(target_position).any():
            self.logger.debug(f"{self.name} ❌ IK target position is NaN — skipping IK")
            return None

        target_position = np.asarray(target_position)
    
        # Handle the case where a full transform matrix was passed in
        if target_position.shape == (4, 4):
            target_position = target_position[:3, 3]  # Extract only position
    
        elif target_position.shape != (3,):
            raise ValueError(f"Expected 3D position or 4x4 pose, got shape {target_position.shape}")
    
        # Now create the full 4x4 target frame
        target_frame = np.eye(4)
        target_frame[:3, 3] = target_position
        
        target_frame = np.eye(4)
        target_frame[2, 3] = 0.5  # just 50 cm straight ahead
    
        self.logger.debug(f"{self.name} IK target position: {target_position}")
        self.logger.debug(f"{self.name} IK target position: {len(self.robot.arm_chain)}, {self.robot.arm_chain}")
        
        self.frontarm_joints = [self.robot.getDevice(arm_name) for arm_name in self.robot.robot_joints.keys()]
              
        current_joint_positions = [0.0] + [
            round(joint.getPositionSensor().getValue(), 2) if hasattr(joint, "getPositionSensor") and joint.getPositionSensor() is not None else 0.0
            for joint in self.frontarm_joints
        ][:8] + [0.0, 0.0] #3.1415, 0, -3.14, 0, 0, 0, 1.57
        self.logger.debug(f"{self.name} IK current positions : {current_joint_positions}")
        
        try:
            ik_result = self.arm_chain.inverse_kinematics(target_position, initial_position=current_joint_positions)
            self.logger.debug(f"{self.name} IK result: {ik_result}")
            return ik_result[1:]
        except Exception as e:
            self.logger.error(f"IK error: {e}")
            return None



    def _move_arm(self, angles):
        for name, angle in zip(self.joint_names, angles):
            motor = self.robot.arm_motors[name]
            motor.setVelocity(1.0)
            motor.setPosition(angle)

    def _arm_reached_target(self):
        threshold = 0.05  # radians
        for name, target_angle in zip(self.joint_names, self.target_angles):
            motor = self.robot.arm_motors[name]
            sensor = motor.getPositionSensor()
            if sensor is None:
                continue
            sensor.enable(self.timestep)
            current_pos = sensor.getValue()
            if abs(current_pos - target_angle) > threshold:
                return False
        return True


class PlaceJar(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_helper, arm_chain, joint_names, gripper_motor, place_pose_angles):
        super(PlaceJar, self).__init__(name)
        self.name = name
        self.robot = robot_helper
        self.arm_chain = arm_chain
        self.joint_names = joint_names
        self.gripper_motor = gripper_motor
        self.place_pose_angles = place_pose_angles
        self.timestep = robot_helper.timestep
        self.stage = "move_arm"

    def update(self):
        if self.stage == "move_arm":
            self._move_arm(self.place_pose_angles)
            if self._arm_reached_target():
                self.stage = "open_gripper"
            return py_trees.common.Status.RUNNING

        elif self.stage == "open_gripper":
            self.gripper_motor.setVelocity(1.0)
            self.gripper_motor.setPosition(1.0)  # open gripper position
            self.stage = "done"
            return py_trees.common.Status.RUNNING

        elif self.stage == "done":
            return py_trees.common.Status.SUCCESS

        else:
            return py_trees.common.Status.SUCCESS

    def _move_arm(self, angles):
        for name, angle in zip(self.joint_names, angles):
            motor = self.robot.arm_motors[name]
            motor.setVelocity(1.0)
            motor.setPosition(angle)

    def _arm_reached_target(self):
        threshold = 0.05  # radians
        for name, target_angle in zip(self.joint_names, self.place_pose_angles):
            motor = self.robot.arm_motors[name]
            sensor = motor.getPositionSensor()
            if sensor is None:
                continue
            sensor.enable(self.timestep)
            current_pos = sensor.getValue()
            if abs(current_pos - target_angle) > threshold:
                return False
        return True
