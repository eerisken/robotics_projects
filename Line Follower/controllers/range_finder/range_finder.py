from controller import Robot, Supervisor, Motor, DistanceSensor
from math import sqrt, atan2, degrees, radians, pi, cos, sin # Added cos, sin for odometry
import numpy as np
from matplotlib import pyplot as plt

# --- Constants ---
TIME_STEP = 64  # milliseconds, standard Webots time step
MAX_SPEED = 6.28  # rad/s, maximum speed of e-puck wheels

# Sensor thresholds (you'll need to tune these in your simulation)
GROUND_SENSOR_DARK_THRESHOLD = 500
GROUND_SENSOR_LIGHT_THRESHOLD = 500

# --- Loop Completion Threshold ---
LOOP_COMPLETION_DISTANCE_THRESHOLD = 0.05 # meters, e.g., 5 cm

# --- e-puck physical constants for odometry ---
# These values are approximate. It's crucial to use the exact dimensions of your e-puck model in Webots.
# You can find them in the robot's .proto file or by inspecting the model in Webots.
WHEEL_RADIUS = 0.0205      # meters (radius of the e-puck wheels)
TRACK_WIDTH = 0.057        # meters (distance between the centers of the two wheels)

# --- Matplotlib Plot Setup (Executed ONCE at the start) ---
plt.ion()
fig, ax = plt.subplots(figsize=(2, 2))

# Lists to store ALL accumulated LiDAR points in world coordinates
all_lidar_x_world = []
all_lidar_z_world = []

# Lists to store the robot's trail (remains the same)
robot_trail_x = []
robot_trail_z = []

ax.set_title("Lidar Scan and Robot Trail in World Coordinates (Mapping)")
ax.set_xlabel("World X-coordinate (m)")
ax.set_ylabel("World Y-coordinate (m)")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

world_plot_limit = 1.0 # Adjust based on your arena size
ax.set_xlim(-world_plot_limit, world_plot_limit)
ax.set_ylim(-world_plot_limit, world_plot_limit)

class EPuckLineFollower:
    def __init__(self):
        self.robot = Supervisor()

        # Ensure the robot is a supervisor (needed for actual position/orientation)
        if not self.robot.getSupervisor():
            print("ERROR: This controller requires the robot to be a Supervisor! Please enable the 'supervisor' field in the robot node properties.")
            self.robot.simulationQuit(1)

        # Get device tags for motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')

        # Set motor velocity control (important for differential drive)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(TIME_STEP)
        self.lidar.enablePointCloud()
        
        self.horizontal_resolution = self.lidar.getHorizontalResolution()
        self.num_layers = self.lidar.getNumberOfLayers()
        
        # --- Angle Calculation based on "rear is 0, clockwise" ---
        self.angle_step_rad = (2 * np.pi) / self.horizontal_resolution
        self.angles_rad_local = np.array([np.pi - (i * self.angle_step_rad) for i in range(self.horizontal_resolution)])
        
        # Get and enable position sensors for odometry (wheel encoders)
        self.left_position_sensor = self.robot.getDevice('left wheel sensor')
        self.right_position_sensor = self.robot.getDevice('right wheel sensor')
        self.left_position_sensor.enable(TIME_STEP)
        self.right_position_sensor.enable(TIME_STEP)
        
        # Store previous encoder readings for calculating wheel displacement
        # Initialize to 0.0, will be updated to actual initial sensor values after first step
        self.prev_left_enc_rad = 0.0
        self.prev_right_enc_rad = 0.0

        # Get device tags for ground sensors
        self.ground_sensor_left = self.robot.getDevice('gs0')
        self.ground_sensor_center = self.robot.getDevice('gs1')
        self.ground_sensor_right = self.robot.getDevice('gs2')

        # Enable ground sensors
        self.ground_sensor_left.enable(TIME_STEP)
        self.ground_sensor_center.enable(TIME_STEP)
        self.ground_sensor_right.enable(TIME_STEP)

        # Get the robot's own Node object (Supervisor access)
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            print("ERROR: Could not get robot's own node as Supervisor!")
            self.robot.simulationQuit(1)

        # For tracking actual (supervisor) position and total distance
        self.initial_position = None # This will be set after the first step
        self.previous_position = None
        self.total_distance_traveled = 0.0

        # For tracking total angular displacement (supervisor)
        self.previous_yaw_radians = None
        self.total_angular_displacement_radians = 0.0

        # --- Odometry Variables (Robot's internal estimate) ---
        # Initialize at (0,0,0) as requested, with 0 orientation (+Z axis forward)
        self.odom_x = 0.0      # Estimated X position (meters)
        self.odom_z = 0.0      # Estimated Z position (meters, Webots' forward/backward axis on ground)
        self.odom_theta = 0.0  # Estimated yaw (orientation) in radians (from +Z towards +X, counter-clockwise)

        self.loop_completed = False
        self.is_first_controller_step = True # Flag to manage first step logic

        print("e-puck line follower (Supervisor) initialized!")
        print(f"Odometry will estimate pose starting from (0,0,0) with initial heading 0 degrees (+Z axis).")

    def read_ground_sensors(self):
        left_val = self.ground_sensor_left.getValue()
        center_val = self.ground_sensor_center.getValue()
        right_val = self.ground_sensor_right.getValue()
        return left_val, center_val, right_val

    def set_motors_speed(self, left_speed, right_speed):
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def get_robot_position(self):
        # Returns the actual position from the simulator (ground truth)
        return self.robot_node.getField('translation').getSFVec3f()

    def get_robot_orientation_yaw_radians(self):
        # Returns the actual yaw from the simulator (ground truth)
        orientation_matrix = self.robot_node.getOrientation()
        yaw_radians = atan2(orientation_matrix[2], orientation_matrix[8]) # R13 and R33 elements
        return yaw_radians

    def calculate_distance(self, pos1, pos2):
        # Calculates Euclidean distance between two 3D points
        return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

    def update_odometry(self):
        """
        Calculates and updates the robot's estimated pose (x, z, theta)
        based on wheel encoder readings.
        """
        # Get current encoder readings (angular position of wheels in radians)
        current_left_enc_rad = self.left_position_sensor.getValue()
        current_right_enc_rad = self.right_position_sensor.getValue()

        # Calculate wheel displacements since the last time step
        delta_left_enc_rad = current_left_enc_rad - self.prev_left_enc_rad
        delta_right_enc_rad = current_right_enc_rad - self.prev_right_enc_rad

        # Calculate linear distance moved by each wheel
        delta_left_m = delta_left_enc_rad * WHEEL_RADIUS
        delta_right_m = delta_right_enc_rad * WHEEL_RADIUS

        # Calculate linear displacement of the robot's center
        delta_s = (delta_left_m + delta_right_m) / 2.0

        # Calculate angular displacement (change in yaw)
        delta_theta = (delta_right_m - delta_left_m) / TRACK_WIDTH

        # Update estimated global pose using differential drive odometry equations
        # Webots: XZ plane, +Z is forward for 0 orientation, rotation around Y.
        # Yaw angle is from +Z towards +X, counter-clockwise.
        
        # Use the average orientation during the step for improved accuracy
        avg_theta_during_step = self.odom_theta + (delta_theta / 2.0)

        # Update X and Z positions
        self.odom_x += delta_s * sin(avg_theta_during_step)
        self.odom_z += delta_s * cos(avg_theta_during_step)

        # Update orientation
        self.odom_theta += delta_theta
        
        # Normalize theta to be within (-pi, pi] for consistency
        self.odom_theta = atan2(sin(self.odom_theta), cos(self.odom_theta))

        # Store current encoder readings for the next time step
        self.prev_left_enc_rad = current_left_enc_rad
        self.prev_right_enc_rad = current_right_enc_rad

    def get_sensor_state(self, sensor_value):
        if sensor_value < GROUND_SENSOR_DARK_THRESHOLD:
            return 'DARK'
        elif sensor_value > GROUND_SENSOR_LIGHT_THRESHOLD:
            return 'LIGHT'
        else:
            return 'MID' # This covers values between thresholds, or exactly on a threshold if not strictly greater/lesser

    # --- Line Following Logic ---
    # This is a simplified proportional control. You'll need to fine-tune it!
    #    
    def line_following_decision(self, sensor_state):
        left_motor_speed = 0
        right_motor_speed = 0
        
        match sensor_state:
            # Case 1: Center sensor on dark (straight line)
            case (_, 'DARK', _): # Wildcard for left and right, center is DARK
                left_motor_speed = MAX_SPEED * 0.8
                right_motor_speed = MAX_SPEED * 0.8
            # Case 2: Left sensor on dark, center light (turn right)
            case ('DARK', 'LIGHT', _): # Left is DARK, Center is LIGHT, Wildcard for right
                left_motor_speed = MAX_SPEED * 0.2
                right_motor_speed = MAX_SPEED * 0.8
            # Case 3: Right sensor on dark, center light (turn left)
            case (_, 'LIGHT', 'DARK'): # Wildcard for left, Center is LIGHT, Right is DARK
                left_motor_speed = MAX_SPEED * 0.8
                right_motor_speed = MAX_SPEED * 0.2
            # Case 4: Lost line (all sensors light)
            case ('LIGHT', 'LIGHT', 'LIGHT'):
                left_motor_speed = MAX_SPEED * 0.5
                right_motor_speed = -MAX_SPEED * 0.5
                print("Line lost! Searching...")
            # Case 5: All sensors dark (very wide line or intersection)
            case ('DARK', 'DARK', 'DARK'):
                left_motor_speed = MAX_SPEED * 0.5
                right_motor_speed = MAX_SPEED * 0.5
                print("All sensors on dark!")
            # Default: Keep going straight, or slightly adjust (any other state)
            case _:
                left_motor_speed = MAX_SPEED * 0.5
                right_motor_speed = MAX_SPEED * 0.5        
        
        return left_motor_speed, right_motor_speed
        
    def transform_lidar_points_to_world(self, local_x, local_z, robot_pose_matrix):
        """
        Transforms 2D LiDAR points (x_local, z_local) from robot's frame to world frame.
        Assumes local_x and local_z are 1D numpy arrays of points.
        robot_pose_matrix is the 4x4 transformation matrix from robot.getPose().
        """
        local_points_homog = np.ones((len(local_x), 4))
        local_points_homog[:, 0] = local_x
        local_points_homog[:, 2] = 0.0      # Assuming 2D LiDAR on X-Z plane
        local_points_homog[:, 1] = local_z
    
        world_points_homog = (robot_pose_matrix @ local_points_homog.T).T
    
        world_x = world_points_homog[:, 0]
        world_z = world_points_homog[:, 1]
    
        return world_x, world_z
        
    def run(self):
        angles = np.linspace(3.1415, -3.1415, 360)
        # Update plot every N simulation steps
        PLOT_UPDATE_EVERY_N_STEPS = 5 # Update plot every 5 steps (adjust as needed)
        step_counter = 0
        MAX_ACCUMULATED_POINTS = 10000
        all_lidar_x_world = np.empty(0)
        all_lidar_z_world = np.empty(0)

        while self.robot.step(TIME_STEP) != -1:
            # --- First step initialization logic ---
            if self.is_first_controller_step:
                # After the very first robot.step(), sensor values are updated.
                # Now, we can safely read them to initialize prev_enc_rad.
                self.prev_left_enc_rad = self.left_position_sensor.getValue()
                self.prev_right_enc_rad = self.right_position_sensor.getValue()
                
                # Also initialize supervisor-based tracking
                current_actual_position = self.get_robot_position()
                current_actual_yaw_radians = self.get_robot_orientation_yaw_radians()
                
                self.initial_position = list(current_actual_position)
                self.previous_position = list(current_actual_position)
                self.previous_yaw_radians = current_actual_yaw_radians
                
                print(f"Initial Supervisor Position: {self.initial_position}")
                print(f"Initial Supervisor Yaw (Orientation): {degrees(self.previous_yaw_radians):.2f} degrees")
                # Odometry is still 0,0,0 as no movement has occurred yet
                print(f"Odometry Initialized: X={self.odom_x:.3f}, Z={self.odom_z:.3f}, Theta={degrees(self.odom_theta):.2f} deg")
                
                self.is_first_controller_step = False # Mark that initialization is done
                
                # For the first step, we just initialized. We don't perform odometry calculation yet,
                # as there's no *previous* movement to calculate a delta from.
                # The first odometry update will happen in the *next* iteration.
                
                # We need to ensure motor speeds are set on the first step too, otherwise robot won't move.
                self.set_motors_speed(MAX_SPEED * 0.5, MAX_SPEED * 0.5) # Set some initial speed to get it moving
                continue # Skip the rest of the loop for this first initialization step

            # --- For all subsequent steps (after the first initialization) ---
            self.update_odometry() # This uses current values (from robot.step() at loop start) and prev_values (from end of last step)

            # --- Get Actual (Supervisor) Pose for comparison/ground truth ---
            current_actual_position = self.get_robot_position()
            current_actual_yaw_radians = self.get_robot_orientation_yaw_radians()

            # --- Calculate total distance traveled (Supervisor) ---
            step_distance = self.calculate_distance(current_actual_position, self.previous_position)
            self.total_distance_traveled += step_distance
            self.previous_position = list(current_actual_position)

            # --- Calculate total angular displacement (Supervisor) ---
            delta_yaw = current_actual_yaw_radians - self.previous_yaw_radians
            if delta_yaw > pi:
                delta_yaw -= 2 * pi
            elif delta_yaw < -pi:
                delta_yaw += 2 * pi
            self.total_angular_displacement_radians += delta_yaw
            self.previous_yaw_radians = current_actual_yaw_radians

            # --- Print current odometry estimates (useful for debugging) ---
            print(f"Odometry Est: X={self.odom_x:.3f}m, Z={self.odom_z:.3f}m, Theta={degrees(self.odom_theta):.2f} deg")


            # --- Check for loop completion (using supervisor's actual position) ---
            distance_from_start = self.calculate_distance(current_actual_position, self.initial_position)
            if distance_from_start < LOOP_COMPLETION_DISTANCE_THRESHOLD and \
               self.robot.getTime() > 2.0: # Give it some time to move away from initial position
                self.loop_completed = True
                print(f"\n--- Loop Completed ---")
                
                # Print Supervisor (Ground Truth) Results
                print(f"Supervisor Final Position: ({current_actual_position[0]:.3f}, {current_actual_position[1]:.3f}, {current_actual_position[2]:.3f})")
                print(f"Supervisor Distance from Start: {distance_from_start:.3f}m")
                print(f"Supervisor Total Track Length: {self.total_distance_traveled:.3f}m")
                print(f"Supervisor Final Orientation: {degrees(current_actual_yaw_radians):.2f} degrees")
                print(f"Supervisor Total Angular Displacement: {degrees(self.total_angular_displacement_radians):.2f} degrees")

                # Print Odometry (Robot's Estimate) Results
                print(f"\nOdometry Final Estimate:")
                print(f"Estimated Position (X, Z): ({self.odom_x:.3f}m, {self.odom_z:.3f}m)")
                print(f"Estimated Orientation (Theta): {degrees(self.odom_theta):.2f} degrees")
                print(f"Estimated Distance from (0,0): {sqrt(self.odom_x**2 + self.odom_z**2):.3f}m")

                self.set_motors_speed(0.0, 0.0) # Stop the robot
                break # Exit the while loop to end the controller script

            # --- Line Following Logic --- (remains the same)
            left_gs, center_gs, right_gs = self.read_ground_sensors()
            sensor_state = (
                self.get_sensor_state(left_gs),
                self.get_sensor_state(center_gs),
                self.get_sensor_state(right_gs)
            )
            # print(f"GS Left: {left_gs:.2f}, GS Center: {center_gs:.2f}, GS Right: {right_gs:.2f}") # Uncomment for debugging
            
            ranges = self.lidar.getRangeImage()
            #print(f"Actual number of data points (len(ranges)): {len(ranges)}")
            
            if ranges and len(ranges) == self.horizontal_resolution * self.num_layers:
                ranges_2d = np.array(ranges[:self.horizontal_resolution])
        
                # Filter out invalid readings
                valid_indices = np.isfinite(ranges_2d)
                valid_ranges = ranges_2d[valid_indices]
                valid_angles = self.angles_rad_local[valid_indices]
        
                # 1. Convert local LiDAR points to robot's local Cartesian frame
                local_x_coords = valid_ranges * np.cos(valid_angles)
                local_z_coords = valid_ranges * np.sin(valid_angles)
        
                # 2. Get robot's current pose
                robot_pose_list = self.robot_node.getPose()
                robot_pose_matrix = np.array(robot_pose_list).reshape((4, 4))
        
                # 3. Transform to world coordinates
                current_scan_world_x, current_scan_world_z = self.transform_lidar_points_to_world(
                    local_x_coords, local_z_coords, robot_pose_matrix
                )
        
                all_lidar_x_world = np.append(all_lidar_x_world, current_scan_world_x)
                all_lidar_z_world = np.append(all_lidar_z_world, current_scan_world_z)
        
                if len(all_lidar_x_world) > MAX_ACCUMULATED_POINTS:
                    excess_points = len(all_lidar_x_world) - MAX_ACCUMULATED_POINTS
                    all_lidar_x_world = all_lidar_x_world[excess_points:]
                    all_lidar_z_world = all_lidar_z_world[excess_points:]
                
                # --- Update Plot ---
                # --- Plot Update Condition ---
                if step_counter % PLOT_UPDATE_EVERY_N_STEPS == 0:
                    ax.clear() # Clear the previous frame for redrawing
            
                    # Redraw all accumulated LiDAR points
                    ax.scatter(all_lidar_x_world, all_lidar_z_world, s=5, c='blue', alpha=0.5, label='Accumulated LiDAR Points')
            
                    # Update robot's trail
                    """
                    robot_position_vec3 = self.robot_node.getPosition()
                    robot_trail_x.append(robot_position_vec3[0])
                    robot_trail_z.append(robot_position_vec3[2])
                    ax.plot(robot_trail_x, robot_trail_z, 'r-', linewidth=1, label='Robot Trail')
            
                    # Redraw current robot position and orientation
                    # The robot's current world X and Z coordinates are robot_position_vec3[0] and robot_position_vec3[2]
                    ax.plot(robot_position_vec3[0], robot_position_vec3[2], 'ro', markersize=10, label='Robot Origin')
            
                    # To draw the robot's forward arrow in world coordinates:
                    # Get robot's orientation (Webots Z is forward if robot is rotated to face X initially, but let's be precise)
                    # The rotation part of robot_pose_matrix is the top-left 3x3
                    rotation_matrix = robot_pose_matrix[0:3, 0:3]
                    # Robot's local forward vector is [1, 0, 0] in its own frame (assuming X is forward)
                    # Transform this local forward vector to world coordinates
                    local_forward_vec = np.array([0.5, 0.0, 0.0]) # Half meter forward along local X-axis
                    # Transform local vector (0, 0.5, 0) to world using only rotation
                    world_forward_vec = (rotation_matrix @ local_forward_vec.reshape(3,1)).flatten()
            
                    ax.arrow(robot_position_vec3[0], robot_position_vec3[2],
                             world_forward_vec[0], world_forward_vec[2], # Only X and Z components for 2D plot
                             head_width=0.2, head_length=0.2, fc='red', ec='red')
                    """ 
                    
                    # IMPORTANT: Restore title, labels, limits, and legend after ax.clear()
                    #ax.set_title("Lidar Scan and Robot Trail in World Coordinates (Mapping)")
                    ax.set_xlabel("World X-coordinate (m)")
                    ax.set_ylabel("World Y-coordinate (m)")
                    ax.set_xlim(-world_plot_limit, world_plot_limit)
                    ax.set_ylim(-world_plot_limit, world_plot_limit)
                    ax.grid(True)
                    ax.set_aspect('equal', adjustable='box')
                    #ax.legend()
            
            
                    # Redraw the canvas
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)
                            
            left_motor_speed, right_motor_speed = self.line_following_decision(sensor_state)
            self.set_motors_speed(left_motor_speed, right_motor_speed)

# Create an instance of the controller and run it
controller = EPuckLineFollower()
controller.run()