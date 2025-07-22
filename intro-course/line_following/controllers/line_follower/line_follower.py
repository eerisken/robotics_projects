from controller import Robot, Motor, DistanceSensor, Supervisor
from math import sqrt, atan2, degrees, radians, pi, sin, cos

# --- Constants ---
TIME_STEP = 64  # milliseconds, standard Webots time step
MAX_SPEED = 6.28  # rad/s, maximum speed of e-puck wheels

# Sensor thresholds (you'll need to tune these in your simulation)
# These values will depend on the light/dark contrast of your texture.
# Typically, a lower value for dark and higher for light.
GROUND_SENSOR_DARK_THRESHOLD = 500  # Example value, adjust as needed
GROUND_SENSOR_LIGHT_THRESHOLD = 500 # Example value, adjust as needed

# --- Loop Completion Threshold ---
# This is the maximum distance the robot can be from its start position
# to be considered "back home". Tune this based on your arena size and desired accuracy.
LOOP_COMPLETION_DISTANCE_THRESHOLD = 0.05 # meters, e.g., 5 cm

# --- e-puck physical constants for odometry ---
# These values are approximate. It's crucial to use the exact dimensions of your e-puck model in Webots.
# You can find them in the robot's .proto file or by inspecting the model in Webots.
WHEEL_RADIUS = 0.0205      # meters (radius of the e-puck wheels)
TRACK_WIDTH = 0.057        # meters (distance between the centers of the two wheels)

class EPuckLineFollower:
    def __init__(self):
        self.robot = Supervisor()

        # Get device tags for motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')

        # Set motor velocity control (important for differential drive)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Get and enable position sensors for odometry (wheel encoders)
        self.left_position_sensor = self.robot.getDevice('left wheel sensor')
        self.right_position_sensor = self.robot.getDevice('right wheel sensor')
        self.left_position_sensor.enable(TIME_STEP)
        self.right_position_sensor.enable(TIME_STEP)
        
        # Store previous encoder readings for calculating wheel displacement
        self.prev_left_enc_rad = 0.0
        self.prev_right_enc_rad = 0.0


        # Get device tags for ground sensors (usually named 'gs0', 'gs1', 'gs2')
        # Check your e-puck node in Webots for exact names.
        self.ground_sensor_left = self.robot.getDevice('gs0') # Assuming these names
        self.ground_sensor_center = self.robot.getDevice('gs1')
        self.ground_sensor_right = self.robot.getDevice('gs2')

        # Enable ground sensors
        self.ground_sensor_left.enable(TIME_STEP)
        self.ground_sensor_center.enable(TIME_STEP)
        self.ground_sensor_right.enable(TIME_STEP)
        
        # Store the initial position of the robot
        # The 'translation' field gives the robot's position (x, y, z)
        # We need to get this AFTER the robot has been initialized by Webots.
        # It's generally best to get this on the first step of the loop.
        self.initial_position = None
        self.previous_position = None # To track distance traveled
        self.total_distance_traveled = 0.0 # Accumulate total distance
        
        # For total angular displacement (rotation)
        self.previous_yaw_radians = None
        self.total_angular_displacement_radians = 0.0
        
        # --- Odometry Variables (Robot's internal estimate) ---
        # Initialize at (0,0,0) as requested, with 0 orientation (+Z axis forward)
        self.odom_x = 0.0      # Estimated X position (meters)
        self.odom_y = 0.0      # Estimated Z position (meters, Webots' forward/backward axis on ground)
        self.odom_omegaz = 0.0  # Estimated yaw (orientation) in radians (from +Z towards +X, counter-clockwise)

        self.loop_completed = False

        print("e-puck line follower initialized!")
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
        # Get the robot's translation field (its position in the world)
        # This returns a list: [x, y, z]
        return self.robot.getSelf().getField('translation').getSFVec3f()

    def calculate_distance(self, pos1, pos2):
        # Calculate Euclidean distance between two 3D points
        return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    
    def get_robot_orientation_yaw_degrees(self):
        # Get the 3x3 rotation matrix (list of 9 floats)
        orientation_matrix = self.robot.getSelf().getOrientation()

        # Extract elements for yaw calculation
        # R11 = orientation_matrix[0]
        R13 = orientation_matrix[2]  # X-component of the forward (Z) vector
        # R31 = orientation_matrix[6]
        R33 = orientation_matrix[8]  # Z-component of the forward (Z) vector

        # Calculate yaw using atan2(sin_theta, cos_theta)
        # Assuming Webots' Z-axis is forward, and rotation around Y is yaw
        yaw_radians = atan2(R13, R33) # Yaw based on forward vector (X, Z components)

        return degrees(yaw_radians)
        
    def get_robot_orientation_yaw_radians(self):
        # Get the 3x3 rotation matrix (list of 9 floats)
        orientation_matrix = self.robot.getSelf().getOrientation()

        # Extract elements for yaw calculation
        # R13 = orientation_matrix[2] (sin_yaw)
        # R33 = orientation_matrix[8] (cos_yaw)
        # Yaw calculated from atan2(sin_yaw, cos_yaw)
        yaw_radians = atan2(orientation_matrix[2], orientation_matrix[8])
        return yaw_radians
    
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
        delta_omega = (delta_right_m - delta_left_m) / TRACK_WIDTH

        # Update estimated global pose using differential drive odometry equations
        # Webots: XZ plane, +Z is forward for 0 orientation, rotation around Y.
        # Yaw angle is from +Z towards +X, counter-clockwise.
        
        # Use the average orientation during the step for improved accuracy
        avg_omegaz_during_step = self.odom_omegaz + (delta_omega / 2.0)

        # Update X and Z positions
        self.odom_x += delta_s * sin(avg_omegaz_during_step)
        self.odom_y += delta_s * cos(avg_omegaz_during_step)

        # Update orientation
        self.odom_omegaz += delta_omega
        
        # Normalize theta to be within (-pi, pi] for consistency
        self.odom_omegaz = atan2(sin(self.odom_omegaz), cos(self.odom_omegaz))

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
    
    def run(self):
        
        # The main simulation loop
        while self.robot.step(TIME_STEP) != -1:
        
            # --- First step initialization logic ---
            if self.is_first_controller_step:
            # Initialize previous encoder readings at the very start of the controller.
            # This prevents large initial delta values if the robot starts with non-zero encoder readings.
            self.prev_left_enc_rad = self.left_position_sensor.getValue()
            self.prev_right_enc_rad = self.right_position_sensor.getValue()
            
            # --- Update Odometry (Robot's Estimate) ---
            # This should be the FIRST thing you do in your loop if you use previous sensor values.
            self.update_odometry()
            
            current_position = self.get_robot_position()
            current_yaw_radians = self.get_robot_orientation_yaw_radians()

            # --- Initialize positions and yaw on the very first step ---
            if self.initial_position is None:
                self.initial_position = list(current_position) # Store a copy
                self.previous_position = list(current_position) # Initialize previous_position for distance
                self.previous_yaw_radians = current_yaw_radians # Initialize previous_yaw for rotation
                print(f"Initial Position: {self.initial_position}")
                print(f"Initial Yaw (Orientation): {degrees(self.previous_yaw_radians):.2f} degrees")
                print(f"Odometry Initialized: X={self.odom_x:.3f}, Z={self.odom_y:.3f}, Theta={degrees(self.odom_omegaz):.2f} deg")
            else:
                # --- Calculate total distance traveled ---
                step_distance = self.calculate_distance(current_position, self.previous_position)
                self.total_distance_traveled += step_distance
                self.previous_position = list(current_position) # Update previous_position

                # --- Calculate total angular displacement (rotation) ---
                delta_yaw = current_yaw_radians - self.previous_yaw_radians

                # Handle angle wrapping (-pi to pi jump)
                # If delta_yaw is > pi, it means it crossed from negative to positive angles (e.g., -170 to 170 means it actually turned +340)
                if delta_yaw > pi:
                    delta_yaw -= 2 * pi
                # If delta_yaw is < -pi, it means it crossed from positive to negative angles (e.g., 170 to -170 means it actually turned -340)
                elif delta_yaw < -pi:
                    delta_yaw += 2 * pi

                self.total_angular_displacement_radians += delta_yaw
                self.previous_yaw_radians = current_yaw_radians # Update previous_yaw

            distance_from_start = self.calculate_distance(current_position, self.initial_position)

            # Check for loop completion (after having moved a bit from start)
            # This 'and' condition prevents immediate stop if it starts very close to the threshold
            if distance_from_start < LOOP_COMPLETION_DISTANCE_THRESHOLD and \
               self.robot.getTime() > 2.0: # Give it some time to move away from initial position
                self.loop_completed = True
                print(f"Loop completed! Returned to initial position. Distance from start: {distance_from_start:.3f}m")
                print(f"Total track length (distance traveled): {self.total_distance_traveled:.3f}m")
                
                # Get and print the final orientation
                #final_orientation_degrees = self.get_robot_orientation_yaw_degrees()
                #print(f"Final Estimated Orientation (Actual from Simulator): {final_orientation_degrees:.2f} degrees")

                # Print Odometry (Robot's Estimate) Results
                print(f"\nOdometry Final Estimate:")
                print(f"Estimated Position (X, Z): ({self.odom_x:.3f}m, {self.odom_y:.3f}m)")
                print(f"Estimated Orientation (Omega): {degrees(self.odom_omegaz):.2f} degrees")
                print(f"Estimated Distance from (0,0): {sqrt(self.odom_x**2 + self.odom_y**2):.3f}m")
                
                self.set_motors_speed(0.0, 0.0) # Stop the robot
                break # Exit the while loop to end the controller script
                
            left_gs, center_gs, right_gs = self.read_ground_sensors()
            sensor_state = (
                self.get_sensor_state(left_gs),
                self.get_sensor_state(center_gs),
                self.get_sensor_state(right_gs)
            )
            # print(f"GS Left: {left_gs:.2f}, GS Center: {center_gs:.2f}, GS Right: {right_gs:.2f}") # Uncomment for debugging

            left_motor_speed, right_motor_speed = self.line_following_decision(sensor_state)
            self.set_motors_speed(left_motor_speed, right_motor_speed)

# Create an instance of the controller and run it
controller = EPuckLineFollower()
controller.run()
