import py_trees
import numpy as np

class TurnToAngle(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_helper, target_angle_deg, angular_speed=0.7):
        super().__init__(name)
        self.robot = robot_helper
        self.timestep = self.robot.timestep
        self.angular_speed = angular_speed
        self.target_angle_deg = target_angle_deg
        self.start_angle = None
        self.done = False

    def initialise(self):
        self.done = False
        self.start_angle = self._get_yaw()
        self.robot.set_wheel_speeds(0.0, 0.0)

    def update(self):
        if self.done:
            return py_trees.common.Status.SUCCESS

        current_yaw = self._get_yaw()
        delta_yaw = self._angle_diff(current_yaw, self.start_angle)

        if abs(delta_yaw) >= abs(np.radians(self.target_angle_deg)):
            self.robot.set_wheel_speeds(0.0, 0.0)
            self.done = True
            return py_trees.common.Status.SUCCESS

        # Choose direction
        sign = np.sign(self.target_angle_deg)
        self.robot.set_wheel_speeds(-sign * self.angular_speed, sign * self.angular_speed)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.robot.set_wheel_speeds(0.0, 0.0)

    def _get_yaw(self):
        imu_vals = self.robot.imu.getRollPitchYaw()
        return imu_vals[2]  # yaw

    def _angle_diff(self, angle1, angle0):
        """Compute signed angle difference from angle0 to angle1."""
        diff = angle1 - angle0
        return (diff + np.pi) % (2 * np.pi) - np.pi
