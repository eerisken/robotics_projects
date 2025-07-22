import time
import numpy as np
import py_trees
import numpy as np
from map_data import Map

class StuckWatchdogWithMapFusion(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, lidar, blackboard, stuck_distance_threshold=0.1, check_interval=3.0):
        """
        :param robot: Webots robot controller (should have get_position() method).
        :param get_lidar_data_fn: Function returning list of (angle, distance) tuples in radians and meters.
        :param map_origin: (origin_x, origin_y) in world coordinates.
        :param map_resolution: map cell size (e.g. 0.05 m/cell).
        :param original_map: The original C-space map (NumPy 2D array).
        :param stuck_distance_threshold: Min movement required to not be considered stuck.
        :param check_interval: Seconds between movement checks.
        """
        super(StuckWatchdogWithMapFusion, self).__init__(name)
        self.name = name
        self.blackboard = blackboard
        self.robot = robot
        self.lidar = lidar
        self.map_origin = (self.blackboard.map_origin_x, self.blackboard.map_origin_y)
        self.map_resolution = self.blackboard.map_resolution
        self.original_map = self.blackboard.c_space_map_object
        self.check_interval = check_interval
        self.stuck_distance_threshold = stuck_distance_threshold

        self.last_pos = None
        self.last_time = time.time()
    
    def initialise(self):
        self.original_map = self.blackboard.c_space_map_object
        
    def update(self):
        if not self.blackboard.stuck_watchdog_enabled:
            return py_trees.common.Status.SUCCESS
            
        current_time = time.time()
        
        #robot_pose = self.blackboard.robot_pose  # Assumes (x, y, theta) or at least (x, y)
        # Get robot pose from blackboard (now updated by UpdateRobotPose behavior)
        robot_x = self.blackboard.robot_pose['x']
        robot_y = self.blackboard.robot_pose['y'] # Using Y from blackboard as Z from GPS in Webots
        robot_theta = self.blackboard.robot_pose['theta']
        robot_pose = (robot_x, robot_y, robot_theta)
        
        current_pos = np.array(robot_pose[:2])

        if self.last_pos is None:
            self.last_pos = current_pos
            self.last_time = current_time
            return py_trees.common.Status.RUNNING

        if current_time - self.last_time >= self.check_interval:
            distance_moved = np.linalg.norm(current_pos - self.last_pos)
            self.last_pos = current_pos
            self.last_time = current_time

            if distance_moved < self.stuck_distance_threshold:
                self.logger.debug(f"{self.name}: ⚠️ Robot appears stuck — fusing LIDAR and triggering replanning")
                self.fuse_lidar_into_map(robot_pose)
                self.blackboard.replan_needed = True

        return py_trees.common.Status.RUNNING

    def fuse_lidar_into_map(self, robot_pose):
        """Fuse current LIDAR scan into the static map to create a dynamic map for replanning."""
        lidar_data = np.array(self.lidar.getRangeImage())  # List of (angle, distance) in robot frame
        lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
        lidar_field_of_view = self.lidar.getFov()
        angles = np.linspace(lidar_field_of_view/2, -lidar_field_of_view/2, lidar_horizontal_resolution)

        for angle, distance in zip(angles, lidar_data):
            if 0.1 < distance < 2.0:  # ignore invalid or too-distant readings
                # LIDAR point in robot frame
                x_robot = distance * np.cos(angle)
                y_robot = distance * np.sin(angle)

                # Transform to world frame
                (robot_x, robot_y, robot_theta) = robot_pose
                print(f"robot_pose: {robot_pose}, type: {type(robot_pose)}")
                
                x_world = robot_x + (x_robot * np.cos(robot_theta) - y_robot * np.sin(robot_theta))
                y_world = robot_y + (x_robot * np.sin(robot_theta) + y_robot * np.cos(robot_theta))

                # Map indices
                map_x = int((x_world - self.map_origin[0]) / self.map_resolution)
                map_y = int((y_world - self.map_origin[1]) / self.map_resolution)

                # Mark obstacle in map
                #if 0 <= map_x < self.original_map.grid.shape[1] and 0 <= map_y < self.original_map.grid.shape[0]:
                #    self.original_map.grid[map_y, map_x] = Map.OCCUPIED  # Mark as OCCUPIED
                    
                self.original_map.set_cell_state(x_world, y_world, Map.OCCUPIED)
