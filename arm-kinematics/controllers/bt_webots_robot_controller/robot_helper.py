from controller import Robot, Supervisor, Camera, Display, Motor, PositionSensor, DistanceSensor, InertialUnit
import numpy as np
from ikpy.chain import Chain
from ikpy.utils import geometry

class RobotHelper(Supervisor):
    def __init__(self):
        super().__init__()
        
        # Time step
        self.timestep = int(self.getBasicTimeStep())
        
        self.robot_joints = {
           'torso_lift_joint' : 0.35,
           'arm_1_joint' : 0.71,
           'arm_2_joint' : 1.02,
           'arm_3_joint' : -2.815,
           'arm_4_joint' : 1.011,
           'arm_5_joint' : 0,
           'arm_6_joint' : 0,
           'arm_7_joint' : 0,
           'gripper_left_finger_joint' : 0,
           'gripper_right__finger_joint': 0,
           'head_1_joint':0,
           'head_2_joint':0
        } 
        
        # Arm joints
        self.arm_joint_names = self.robot_joints.keys()
        
        self.arm_joints = [self.getDevice(f'arm_{i}_joint') for i in range(1, 8)]
        self.torso_joint = self.getDevice('torso_lift_joint')
        self.head_2_joint = self.getDevice('head_2_joint')

        self.arm_motors = {name: self.getDevice(name) for name in self.arm_joint_names}
        self.arm_sensors = {name + "_sensor": self.getDevice(name + "_sensor") for name in self.arm_joint_names if self.getDevice(name + "_sensor")}
        self.arm_chain = Chain.from_urdf_file("only_arm.urdf", base_elements=["torso_lift_link", "torso_lift_link_TIAGo front arm_joint"])

        #for motor in self.arm_motors.values():
        #    motor.setVelocity(1.0)

        # Head joints (if needed)
        self.head_motors = {
            "head_1_joint": self.getDevice("head_1_joint"),
            "head_2_joint": self.getDevice("head_2_joint"),
        }

        # Wheels
        self.left_motor = self.getDevice("wheel_left_joint")
        self.right_motor = self.getDevice("wheel_right_joint")

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Display (for image output)
        try:
            self.display = self.getDevice("display")
        except:
            self.display = None

        # IMU
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        # LIDAR
        self.lidar = self.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # GPS / Compass
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        # Gripper (if available)
        self.gripper_motors = {}
        for name in ["gripper_left_finger_joint", "gripper_right_finger_joint"]:
            try:
                self.gripper_motors[name] = self.getDevice(name)
                self.gripper_motors[name].setVelocity(0.03)
            except:
                pass
                
        self.lidar_fov = self.lidar.getFov()
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_angle_step = self.lidar_fov / self.lidar_resolution
        
    def get_camera_image(self):
        image = self.camera_rgb.getImage()
        w, h = self.camera_width, self.camera_height
        img = np.frombuffer(image, np.uint8).reshape((h, w, 4))[:, :, :3]  # Drop alpha
        
        return img

    def get_camera_pose(self):
        camera_node = self.getFromDef("CAMERA")

        if camera_node is None:
            print("Camera node with DEF name 'Astra' not found!")
            return None
        else:
            pos = camera_node.getField('translation').getSFVec3f()
            rot = camera_node.getField('rotation').getSFRotation()
            
            return {"translation":pos, "rotation":rot}
    
    def set_display_image(self, image):
        from PIL import Image
        
        if self.display is None:
            return
            
        # Convert RGB → BGR manually
        image_bgr = image[..., ::-1]  # (H, W, 3)
    
        img = Image.fromarray(image_bgr)
        img = img.resize((self.display.getWidth(), self.display.getHeight()))
    
        bgr_bytes = img.tobytes()
        handle = self.display.imageNew(bgr_bytes, self.display.RGB, img.width, img.height)
        self.display.imagePaste(handle, 0, 0, False)

    def get_lidar_range_image(self):
        ranges = np.array(self.lidar.getRangeImage())
        angles = np.linspace(self.lidar_fov/2, -self.lidar_fov/2, self.lidar_resolution)
        return ranges, angles

    def set_wheel_speeds(self, v_left, v_right):
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)
        
    def set_wheel_positions(self, left_pos, right_pos):
        self.left_motor.setPosition(left_pos)
        self.right_motor.setPosition(right_pos)

    def get_end_effector_pose(self):
        joint_positions = [self.arm_motors[j].getTargetPosition() for j in self.arm_joint_names]
        T = self.arm_chain.forward_kinematics(joint_positions)
        return T  # 4x4 homogeneous transform in base frame

    def camera_pixel_to_world(self, u, v):
        if self.camera_depth is None:
            return None  # Depth camera not available
            
        if u < 0 or v < 0 or u >= self.camera_depth_width or v >= self.camera_depth_height:
            return None
        
        Z = self.camera_depth.imageGetDepth(self.camera_depth.getImage(), self.camera_depth_width, u, v)

        if Z == float('inf') or Z == 0:
            return None
        
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        camera_point = np.array([X, Y, Z, 1.0])
        T = self.get_camera_transform()
        world_point = T @ camera_point
        return world_point[:3]

    def get_camera_transform(self):
        # Hardcoded camera pose relative to base, or retrieve dynamically
        T = np.identity(4)
        T[0:3, 3] = [0.2, 0.0, 1.2]  # example translation
        return T

    def lidar_points_to_world(self):
        ranges, angles = self.get_lidar_range_image()
        points = []
        for r, theta in zip(ranges, angles):
            if r == float('inf'):
                continue
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            point = np.array([x, y, 0.0, 1.0])
            T = self.get_lidar_transform()
            world_point = T @ point
            points.append(world_point[:3])
        return np.array(points)

    def get_lidar_transform(self):
        # Hardcoded transform lidar-to-base or extract from URDF
        T = np.identity(4)
        T[0:3, 3] = [0.2, 0.0, 0.8]  # example offset
        return T
        
    def set_arm_positions(self, positions):
        for name, pos in zip(self.arm_joint_names, positions):
            self.arm_motors[name].setPosition(pos)
    
    def get_arm_positions(self):
        return [self.arm_sensors[name + "_sensor"].getValue() for name in self.arm_joint_names]

    def open_gripper(self):
        for name in self.gripper_motors:
            self.gripper_motors[name].setPosition(0.045)  # adjust limits
    
    def close_gripper(self):
        for name in self.gripper_motors:
            self.gripper_motors[name].setPosition(0.0)
        
        