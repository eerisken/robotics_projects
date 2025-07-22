import py_trees
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class PickJar(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, blackboard, y_tolerance=0.005, x_target=0.9, grip_force_threshold=30.0):
        super().__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.y_tolerance = y_tolerance
        self.x_target = x_target
        self.grip_force_threshold = grip_force_threshold
        self.state = "idle"

    def setup(self):
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(32)
        self.camera.recognitionEnable(32)

        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.gripper_left = self.robot.getDevice('gripper_left_finger_joint')
        self.gripper_right = self.robot.getDevice('gripper_right_finger_joint')
        self.torso_joint = self.robot.getDevice('torso_lift_joint')
        self.torso_sensor = self.torso_joint.getPositionSensor()
        self.torso_sensor.enable(32)

        self.arm_4_joint = self.robot.getDevice('arm_4_joint')
        self.arm_4_sensor = self.arm_4_joint.getPositionSensor()
        self.arm_4_sensor.enable(32)

        self.gripper_left.enableForceFeedback(32)
        self.gripper_right.enableForceFeedback(32)

        # Velocity setup
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        self.gripper_left.setVelocity(0.03)
        self.gripper_right.setVelocity(0.03)
        self.torso_joint.setVelocity(0.05)
        self.arm_4_joint.setVelocity(0.5)

    def initialise(self):
        self.state = "aligning"
        self.feedback_message = "Starting alignment and approach"
        self.gripper_left.setPosition(0.045)
        self.gripper_right.setPosition(0.045)
        self.torso_joint.setPosition(0.3)
        self.lift_start_time = 0
        self.back_start_time = 0
        self.back_duration = 7.0

    def update(self):
        objects = self.camera.getRecognitionObjects()
        if not objects:
            self.logger.debug(f"{self.name} No objects detected.")
            return py_trees.common.Status.FAILURE

        obj = objects[0]
        x_error = obj.getPosition()[0]
        y_error = obj.getPosition()[1]
        error_to_target = y_error - 0.02

        # FSM logic
        if self.state == "aligning":
            if abs(error_to_target) > self.y_tolerance:
                turn_speed = 0.5 * np.sign(error_to_target)
                self.left_motor.setVelocity(-turn_speed)
                self.right_motor.setVelocity(turn_speed)
                return py_trees.common.Status.RUNNING

            if x_error > self.x_target:
                forward_speed = 0.5
                self.left_motor.setVelocity(forward_speed)
                self.right_motor.setVelocity(forward_speed)
                return py_trees.common.Status.RUNNING

            # Stop and start gripping
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            self.gripper_left.setPosition(0.0)
            self.gripper_right.setPosition(0.0)
            self.state = "closing_gripper"
            self.feedback_message = "Closing gripper"
            return py_trees.common.Status.RUNNING

        elif self.state == "closing_gripper":
            force = abs(self.gripper_left.getForceFeedback() + self.gripper_right.getForceFeedback())
            if force > self.grip_force_threshold:
                self.torso_joint.setPosition(0.35)
                self.state = "raising_torso"
            return py_trees.common.Status.RUNNING

        elif self.state == "raising_torso":
            if abs(self.torso_sensor.getValue() - 0.35) < 0.01:
                self.back_start_time = self.robot.getTime()
                self.state = "moving_back"
            return py_trees.common.Status.RUNNING

        elif self.state == "moving_back":
            if self.robot.getTime() - self.back_start_time < self.back_duration:
                self.left_motor.setVelocity(-0.5)
                self.right_motor.setVelocity(-0.5)
            else:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                self.arm_4_joint.setPosition(1.0)
                self.state = "lifting_arm"
                self.lift_start_time = self.robot.getTime()
            return py_trees.common.Status.RUNNING

        elif self.state == "lifting_arm":
            # Optionally wait or monitor arm_4_sensor
            self.back_duration = 6.0
            print("[Motion] Arm lifted complete")
            self.feedback_message = "Arm lifted, task complete"
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

class PlaceJar(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, blackboard):
        super().__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.yaw_tolerance = 0.05
        self.state = "idle"

    def setup(self, **kwargs):
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.gps = self.robot.getDevice('gps')
        self.compass = self.robot.getDevice('compass')
        self.gps.enable(32)
        self.compass.enable(32)

        self.arm_4_joint = self.robot.getDevice('arm_4_joint')
        self.arm_4_joint.setVelocity(0.5)
        self.arm_4_sensor = self.arm_4_joint.getPositionSensor()
        self.arm_4_sensor.enable(32)

        self.gripper_left = self.robot.getDevice('gripper_left_finger_joint')
        self.gripper_right = self.robot.getDevice('gripper_right_finger_joint')
        self.gripper_left.setVelocity(0.03)
        self.gripper_right.setVelocity(0.03)
        self.gripper_left.getPositionSensor().enable(32)
        self.gripper_right.getPositionSensor().enable(32)

        self.torso_joint = self.robot.getDevice('torso_lift_joint')
        self.torso_joint.getPositionSensor().enable(32)

    def initialise(self):
        self.state = "rotate_to_target"
        self.gripper_wait_start = None

        self.arm_4_joint.setPosition(1.75)  # lift up first
        compass = self.compass.getValues()
        self.initial_heading = np.arctan2(compass[0], compass[1])

        target = self.blackboard.goal_table
        if target is None:
            self.logger.error("PlaceJar: No place target on blackboard!")
            self.target_x, self.target_y = None, None
        else:
            self.target_x, self.target_y = target

    def update(self):
        # Get current state
        state = self.state
        robot_time = self.robot.getTime()
        gps = self.gps.getValues()
        compass = self.compass.getValues()

        x, y = gps[0], gps[1]
        heading = np.arctan2(compass[0], compass[1])

        dx = self.target_x - x
        dy = self.target_y - y
        target_angle = np.arctan2(dy, dx)
        angle_diff = (target_angle - heading + np.pi) % (2 * np.pi) - np.pi
        back_angle_diff = (self.initial_heading - heading + np.pi) % (2 * np.pi) - np.pi

        # FSM state machine
        if state == "rotate_to_target":
            if abs(angle_diff) > self.yaw_tolerance:
                speed = 0.2 * np.sign(angle_diff)
                self.left_motor.setVelocity(-speed)
                self.right_motor.setVelocity(speed)
            else:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                self.torso_joint.setPosition(0.15)
                self.arm_4_joint.setPosition(0.0)
                self.state = "drive_to_target"
            return py_trees.common.Status.RUNNING

        elif state == "drive_to_target":
            dist = np.hypot(dx, dy)
            if dist > 0.55:
                self.left_motor.setVelocity(0.2)
                self.right_motor.setVelocity(0.2)
            else:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                self.state = "open_gripper"
            return py_trees.common.Status.RUNNING

        elif state == "open_gripper":
            arm_pos = self.arm_4_sensor.getValue()
            if abs(arm_pos - 0.0) < 0.02:
                self.gripper_left.setPosition(0.045)
                self.gripper_right.setPosition(0.045)
                if self.gripper_wait_start is None:
                    self.gripper_wait_start = robot_time
                elif robot_time - self.gripper_wait_start >= 2.0:
                    self.arm_4_joint.setPosition(1.75)
                    self.state = "reset_arm"
            return py_trees.common.Status.RUNNING

        elif state == "reset_arm":
            if abs(self.arm_4_sensor.getValue() - 1.75) < 0.02:
                self.state = "rotate_back"
            return py_trees.common.Status.RUNNING

        elif state == "rotate_back":
            if abs(back_angle_diff) > self.yaw_tolerance:
                speed = 0.5 * np.sign(back_angle_diff)
                self.left_motor.setVelocity(-speed)
                self.right_motor.setVelocity(speed)
            else:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                self.arm_4_joint.setPosition(0)
                self.torso_joint.setPosition(0.3)
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.arm_4_joint.setPosition(1.75)
        self.state = "idle"
