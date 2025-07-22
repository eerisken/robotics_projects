# mapping.py
import py_trees
import numpy as np
import os
from controller import LidarPoint
import struct
import controller
from pid import PID
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter # Added for noise reduction

# Import the new Map, CSpaceGenerator, and TrajectoryFollower classes
from map_data import Map
from c_space import CSpaceGenerator
from pid import TrajectoryFollower # Import the new class

# --- Global Constants (moved into Probabilistic_Occupancy_Map or passed to it) ---
TIME_STEP = 32  # milliseconds
MAX_SPEED = 6.28  # rad/s (Taigolite's max speed)
       
class MapEnvironment(py_trees.behaviour.Behaviour):
    """
    A behaviour that continuously maps the environment using Lidar, GPS, and Compass.
    It builds an occupancy grid and generates a configuration space map.
    This behaviour never yields SUCCESS and is designed to run indefinitely
    until its parent (a Parallel node) terminates it.
    """  
    def __init__(self, name, robot, lidar, gps, compass, blackboard):
        super(MapEnvironment, self).__init__(name)
        self.robot = robot
        self.lidar = lidar
        self.gps = gps
        self.compass = compass
        self.blackboard = blackboard
        
        # Get the Map object from blackboard
        self.map = self.blackboard.map

        self.logger.debug(f"{self.name}: Initialized.")


    def initialise(self):
        """Initializes the Lidar, GPS, and Compass devices."""
        #self.lidar.enable(self.robot.getBasicTimeStep())
        #self.lidar.enablePointCloud()
        #self.gps.enable(self.robot.getBasicTimeStep())
        #self.compass.enable(self.robot.getBasicTimeStep())
        self.logger.debug(f"{self.name}: Initialized.")

    def update(self):
        """
        Collects sensor data, updates the occupancy grid, and stores it on the blackboard.
        This method always returns RUNNING as mapping is a continuous process.
        """
        # Get robot pose from blackboard (now updated by UpdateRobotPose behavior)
        robot_x = self.blackboard.robot_pose['x']
        robot_y = self.blackboard.robot_pose['y'] # Using Y from blackboard as Z from GPS in Webots
        robot_theta = self.blackboard.robot_pose['theta']
        
        # Get Lidar data
        # Lidar returns range image (distances) and point cloud (x,y,z points relative to lidar).
        # Using point cloud is generally easier for mapping.
        ranges = np.array(self.lidar.getRangeImage())
        lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
        lidar_points = self.lidar.getNumberOfPoints()
        lidar_field_of_view = self.lidar.getFov()
        lidar_max_range = self.lidar.getMaxRange()
        
        """
        # Get GPS
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]   
        # Read compass values
        robot_angle = np.arctan2(self.compass.getValues()[0],self.compass.getValues()[1])
        # Read lidar data
        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges == np.inf] = 100
        # Transform matrix 
        w_T_r = np.array([[np.cos(robot_angle), -np.sin(robot_angle), xw], 
                        [np.sin(robot_angle), np.cos(robot_angle), yw],
                        [0, 0, 1]])        
        # Add lidar offset from robot center (0.202 m)
        lidar_offset = np.array([[1, 0,  0.202],
                                    [0, 1,  0],
                                    [0, 0,  1]])               
        X_i = np.array([ranges*np.cos(self.angles), ranges*np.sin(self.angles), np.ones((667,))])
        D = w_T_r @ (lidar_offset @ X_i)
        """
        
        # Define the number of readings to exclude from each end
        EXCLUDE_READINGS = 80

        w_T_r = np.array([[np.cos(robot_theta), -np.sin(robot_theta), robot_x], 
                 [np.sin(robot_theta), np.cos(robot_theta), robot_y],
                 [0,0,1]])
                 
        angles = np.linspace(lidar_field_of_view/2, -lidar_field_of_view/2, lidar_horizontal_resolution)[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        ranges = np.minimum(ranges, 7.0)[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        
        # Add lidar offset from robot center (0.202 m)
        lidar_offset = np.array([[1, 0,  0.202],
                                    [0, 1,  0],
                                    [0, 0,  1]]) 
                                    
        x_i = np.array([ranges*np.cos(angles), ranges*np.sin(angles), np.ones(ranges.shape[0])])
        D = w_T_r @ x_i #D = w_T_r @ (lidar_offset @ x_i) 
        
        for k in range(D.shape[1]):
            lx, ly = D[0, k], D[1, k]
            # Set cell state in the Map object
            self.map.set_cell_state(lx, ly, Map.OCCUPIED)
        
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Saves the generated map and computes the configuration space map
        when the behaviour is terminated.
        """
        if new_status == py_trees.common.Status.SUCCESS:
            print(f"MapEnvironment: Terminating due to parent success. Saving map.")
        else:
            print(f"MapEnvironment: Terminating with status {new_status}. Saving map.")

        # Generate C-space using the CSpaceGenerator
        # Assuming robot radius is 0.265m, and map_resolution is 0.05m/px
        # robot_radius_pixels = 0.265 / 0.05 = 5.3 -> round up to 6 pixels for dilation
        robot_radius_pixels_val = 5
        filter_size_val = 4 # Example filter size

        c_space_gen = CSpaceGenerator(self.map, 
                                      robot_radius_pixels=robot_radius_pixels_val,
                                      filter_size=filter_size_val)
        c_space_gen.generate_c_space()

        # Save and plot both original map and C-space map
        c_space_gen.save_and_plot_maps(
            map_file_name="kitchen_map.npy",
            c_space_file_name="kitchen_c_space.npy",
            map_plot_name="final_occupancy_map.png",
            c_space_plot_name="final_c_space_map.png"
        )
        
        # Store the generated C-space Map object on the blackboard
        self.blackboard.c_space_map_object = c_space_gen.c_space_map_object
            
        self.logger.debug(f"{self.name}: Terminated.")

class MoveAroundTable(py_trees.behaviour.Behaviour):
    """
    A behaviour that moves the robot around a predefined set of waypoints
    to facilitate mapping. This behaviour yields SUCCESS once all waypoints
    are visited.
    """
    def __init__(self, name, robot, gps, compass, left_motor, right_motor, blackboard): 
        super(MoveAroundTable, self).__init__(name)
        self.robot = robot
        self.gps = gps
        self.compass = compass
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.blackboard = blackboard
        #self.waypoints = [] # List of (x, y) global coordinates
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.35 # Meters
        self.move_speed = 0.5 # Radians/sec
        
        # Initialize TrajectoryFollower for simple mapping movement
        # PID gains are not used in calculate_wheel_speeds_simple
        self.trajectory_follower = TrajectoryFollower(
            linear_speed=0.5, # Meters/sec (approximate)
            angular_speed_max=0.8, # Radians/sec (max angular speed for turning)
            waypoint_tolerance=0.35, # Meters
            pure_turn_angle_threshold=np.deg2rad(15), # Radians (e.g., 15 degrees)
            kp=2.4, ki=0.0, kd=0.05 # PID gains for heading control
        )
        
        # Get mapping parameters from blackboard
        self.map_resolution = self.blackboard.map_resolution
        self.map_origin_x = self.blackboard.map_origin_x
        self.map_origin_y = self.blackboard.map_origin_y
        self.map_x_dim_meters = self.blackboard.map_x_dim_meters
        self.map_y_dim_meters = self.blackboard.map_y_dim_meters
        #self.map_dim_pixels = self.blackboard.map_dim_pixels # This is now a tuple (rows, cols)
        
        self.current_waypoint_idx = 0
        self.dt = self.robot.getBasicTimeStep() / 1000.0 # Convert ms to seconds
        
        self.time_step = int(self.robot.getBasicTimeStep())
        self.heading_pid = PID(kp=2.5, ki=0.00, kd=0.1)

        # Define specific waypoints around a typical kitchen table (adjust as needed for your world)
        # These are relative to the center of the world (0,0,0)
        # Assuming the table is somewhat centered around (0,0)
        self.waypoints = self.blackboard.waypoints_to_follow
        #self.waypoints = self.waypoints + list(reversed(self.waypoints))
        self.logger.debug(f"{self.name}: Initialized with waypoints: {self.waypoints}.")

    def initialise(self):
        """Resets the waypoint index and enables motors."""
        self.current_waypoint_idx = 0
        #self.trajectory_follower.reset_pid()
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug(f"{self.name}: Initialized.")
        
    def update(self):
        """
        Moves the robot towards the current waypoint.
        Returns SUCCESS if all waypoints are visited.
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            self.logger.debug(f"{self.name}: All waypoints visited. Success!")
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            return py_trees.common.Status.SUCCESS

        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        current_robot_pose = self.blackboard.robot_pose # Use the updated pose

        if self.trajectory_follower.is_waypoint_reached(current_robot_pose, [target_x, target_y]):
            self.logger.info(f"{self.name}: Reached mapping waypoint {self.current_waypoint_idx}: ({target_x:.2f}, {target_y:.2f})")
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.info(f"{self.name}: Moving to next mapping waypoint {self.current_waypoint_idx}: {self.waypoints[self.current_waypoint_idx]}")
                # Reset PID if TrajectoryFollower had one, though simple control doesn't use it
                self.trajectory_follower.reset_pid() 
                return py_trees.common.Status.RUNNING # Continue to next waypoint

        # Calculate wheel speeds using the simple method for mapping
        left_speed, right_speed = self.trajectory_follower.calculate_wheel_speeds_pid(
            current_robot_pose, [target_x, target_y], self.dt
        )
        self.logger.debug(f"{self.name}: Left Speed, Right Speed: ({left_speed:.2f}, {right_speed:.2f}).")
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
                
        self.logger.debug(f"{self.name}: Navigating to ({target_x:.2f}, {target_y:.2f}). Current pose: ({current_robot_pose['x']:.2f}, {current_robot_pose['y']:.2f}, {np.degrees(current_robot_pose['theta']):.2f}°).")
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Stops the robot motors when the behaviour terminates."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug(f"{self.name}: Terminated with status {new_status}.")

