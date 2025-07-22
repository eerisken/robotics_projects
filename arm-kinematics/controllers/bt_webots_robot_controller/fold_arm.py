import py_trees
import numpy as np
from ikpy.chain import Chain
from controller import Robot

class FoldArm(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, blackboard, urdf_path="tiago.urdf"):
        super(FoldArm, self).__init__(name)
        self.robot = robot
        self.blackboard = blackboard

        self.home_pose = {
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
        self.initialised = False

    def initialise(self):
        if not self.initialised:
            for joint in self.robot.arm_joints:
                joint.getPositionSensor().enable(32)
                joint.setVelocity(0.5)
            self.robot.torso_joint.getPositionSensor().enable(32)
            self.robot.torso_joint.setVelocity(0.07)
    
            # Define target positions
            self.target_positions = [3.1415, 0, -3.14, 0, 0, 0, 1.57]
            self.torso_target = 0.3
    
            self.tolerance = 0.02  # rad tolerance
            self.head2_target = -0.3
            self.tolerance = 0.05  # rad or meters
            self.robot.getDevice('wheel_left_joint').setVelocity(0.0)
            self.robot.getDevice('wheel_right_joint').setVelocity(0.0)
            
            for joint, pos in zip(self.robot.arm_joints, self.target_positions):
                joint.setPosition(pos)
            self.robot.torso_joint.setPosition(self.torso_target)
            self.robot.head_2_joint.setPosition(self.head2_target)
            self.initialised = True

    def update(self):
        # Optionally check convergence
        arm_folded = True
        for joint, target in zip(self.robot.arm_joints, self.target_positions):
            actual = joint.getPositionSensor().getValue()
            if abs(actual - target) > self.tolerance:
                arm_folded = False
    
        torso_actual = self.robot.torso_joint.getPositionSensor().getValue()
        if abs(torso_actual - self.torso_target) > self.tolerance:
            arm_folded = False
    
        if arm_folded:     
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING
