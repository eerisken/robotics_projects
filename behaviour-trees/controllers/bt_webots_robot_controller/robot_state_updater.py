# robot_state_updater.py
import py_trees
import numpy as np

class UpdateRobotPose(py_trees.behaviour.Behaviour):
    """
    A behaviour that continuously updates the robot's pose (x, y, theta)
    on the blackboard using GPS and Compass sensor data.
    This behaviour always returns SUCCESS as its primary role is to update state.
    """
    def __init__(self, name, robot, gps, compass, blackboard):
        super(UpdateRobotPose, self).__init__(name)
        self.robot = robot
        self.gps = gps
        self.compass = compass
        self.blackboard = blackboard
        self.logger.debug(f"{self.name}: Initialized.")

    def initialise(self):
        """Enables GPS and Compass sensors."""
        # Sensors should ideally be enabled once at the start of the controller,
        # but enabling here ensures they are active if this behavior is used.
        self.gps.enable(int(self.robot.getBasicTimeStep())) #(int(self.blackboard.map.map_resolution * 1000)) # Use map resolution to derive time step for GPS
        self.compass.enable(int(self.robot.getBasicTimeStep())) #(int(self.blackboard.map.map_resolution * 1000)) # Use map resolution to derive time step for Compass
        self.logger.debug(f"{self.name}: Initialised.")

    def update(self):
        """
        Reads GPS and Compass data and updates the robot's pose on the blackboard.
        """
        gps_values = self.gps.getValues()
        compass_values = self.compass.getValues()

        robot_x = gps_values[0]
        robot_y = gps_values[1] # Using Z from GPS as Y in 2D map for horizontal plane
        robot_theta = np.arctan2(compass_values[0], compass_values[1]) # Yaw in radians

        self.blackboard.robot_pose['x'] = robot_x
        self.blackboard.robot_pose['y'] = robot_y
        self.blackboard.robot_pose['theta'] = robot_theta

        # self.logger.debug(f"{self.name}: Pose updated to X={robot_x:.2f}, Y={robot_y:.2f}, Theta={np.degrees(robot_theta):.2f}°")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """Does nothing on termination as it's a continuous update behaviour."""
        self.logger.debug(f"{self.name}: Terminated with status {new_status}.")