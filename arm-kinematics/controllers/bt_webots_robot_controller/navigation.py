# navigation.py
import py_trees
import numpy as np
import struct
import controller
from pid import PID

# Import the new TrajectoryFollower class
from pid import TrajectoryFollower

# --- Global Constants (moved into Probabilistic_Occupancy_Map or passed to it) ---
TIME_STEP = 32  # milliseconds
MAX_SPEED = 6.28  # rad/s (Taigolite's max speed)
WAYPOINT_THRESHOLD = 0.35  # meters, distance to consider waypoint reached

class NavigateToWaypoints(py_trees.behaviour.Behaviour):
    """
    A behaviour that navigates the robot along a list of waypoints provided
    via the blackboard.
    """
    def __init__(self, name, robot, gps, compass, left_motor, right_motor, blackboard):
        super(NavigateToWaypoints, self).__init__(name)
        self.robot = robot
        self.gps = gps
        self.compass = compass
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.blackboard = blackboard
        
        self.current_waypoint_idx = 0
        
        self.waypoint_tolerance = 0.35 # Meters
        self.linear_speed = 0.5 # Meters/sec (approximate)
        self.angular_speed_max = 0.8 # Radians/sec

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Initialize TrajectoryFollower with PID gains for navigation
        self.trajectory_follower = TrajectoryFollower(
            linear_speed=0.5, # Meters/sec (approximate)
            angular_speed_max=0.8, # Radians/sec (max angular speed for turning)
            waypoint_tolerance=0.35, # Meters
            pure_turn_angle_threshold=np.deg2rad(15), # Radians (e.g., 15 degrees)
            kp=2.0, ki=0.0, kd=0.03 # PID gains for heading control
        )
        self.dt = self.robot.getBasicTimeStep() / 1000.0 # Convert ms to seconds
        
        self._success = False
        
        self.logger.debug(f"{self.name}: Initialized.")

    def initialise(self):
        if self._success:
            return  # Skip re-initialization if already completed
            
        self.current_waypoint_idx = 0
        self.trajectory_follower.reset_pid() # Reset PID state for new navigation task
        
        if not self.blackboard.waypoints_to_follow:
            self.logger.warning(f"{self.name}: No waypoints to follow. Failing.")
            return py_trees.common.Status.FAILURE
        self.logger.info(f"{self.name}: Starting navigation with {len(self.blackboard.waypoints_to_follow)} waypoints.")
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug(f"{self.name}: Initialised.")
     
    def update(self):
        """
        Moves the robot towards the current waypoint using TrajectoryFollower.
        Returns SUCCESS if all waypoints are visited.
        """
        if self._success:
            return py_trees.common.Status.SUCCESS
            
        if not self.blackboard.waypoints_to_follow:
            self.logger.warning(f"{self.name}: Waypoints list became empty during navigation. Failing.")
            return py_trees.common.Status.FAILURE

        # Check if current waypoint is reached first
        if self.current_waypoint_idx >= len(self.blackboard.waypoints_to_follow):
            self.logger.info(f"{self.name}: All waypoints visited. Success!")
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            self._success = True
            self.blackboard.stuck_watchdog_enabled = False
            return py_trees.common.Status.SUCCESS

        target_x, target_y = self.blackboard.waypoints_to_follow[self.current_waypoint_idx]
        current_robot_pose = self.blackboard.robot_pose # Use the updated pose

        # Log current state for debugging
        self.logger.debug(f"{self.name}: Current Pose: X={current_robot_pose['x']:.2f}, Y={current_robot_pose['y']:.2f}, Theta={np.degrees(current_robot_pose['theta']):.2f}°")
        self.logger.debug(f"{self.name}: Target Waypoint: X={target_x:.2f}, Y={target_y:.2f}")

        if self.trajectory_follower.is_waypoint_reached(current_robot_pose, [target_x, target_y]):
            self.logger.info(f"{self.name}: Reached waypoint {self.current_waypoint_idx}: ({target_x:.2f}, {target_y:.2f})")
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.blackboard.waypoints_to_follow):
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.info(f"{self.name}: Moving to next waypoint {self.current_waypoint_idx}: {self.blackboard.waypoints_to_follow[self.current_waypoint_idx]}")
                self.trajectory_follower.reset_pid() # Reset PID for the new segment
                return py_trees.common.Status.RUNNING # Continue to next waypoint
        
        # Calculate wheel speeds using the PID-based method for navigation
        left_speed, right_speed = self.trajectory_follower.calculate_wheel_speeds_pid(
            current_robot_pose, [target_x, target_y], self.dt
        )
        
        # Log calculated speeds for debugging
        self.logger.debug(f"{self.name}: Calculated Speeds: Left={left_speed:.2f}, Right={right_speed:.2f}")

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        self.logger.debug(f"{self.name}: Navigating to ({target_x:.2f}, {target_y:.2f}). Current pose: ({current_robot_pose['x']:.2f}, {current_robot_pose['y']:.2f}, {np.degrees(current_robot_pose['theta']):.2f}°).")
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Stops the robot motors when the behaviour terminates."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug(f"{self.name}: Terminated with status {new_status}.")
        
