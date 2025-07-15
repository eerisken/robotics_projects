import numpy as np

MAX_SPEED = 6.28  # rad/s (Taigolite's max speed)

# --- PID Controller Class ---
class PID:
    """
    A simple PID controller.
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """
        Updates the PID controller and returns the control output.
        :param error: The current error value.
        :param dt: The time step since the last update.
        :return: The PID control output.
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        """Resets the integral and previous error terms of the PID controller."""
        self.integral = 0.0
        self.prev_error = 0.0
        
class TrajectoryFollower:
    """
    A class to handle the movement logic for a differential drive robot,
    calculating wheel speeds to follow a trajectory or reach a waypoint.
    It incorporates a PID controller for heading control.
    """
    def __init__(self, linear_speed: float, angular_speed_max: float, 
                 waypoint_tolerance: float, pure_turn_angle_threshold: float,
                 kp: float, ki: float, kd: float):
        """
        Initializes the TrajectoryFollower.

        :param linear_speed: Desired linear speed of the robot (m/s).
        :param angular_speed_max: Maximum angular speed for turning (rad/s).
        :param waypoint_tolerance: Distance threshold to consider a waypoint reached (m).
        :param pure_turn_angle_threshold: Angular threshold (rad) to decide between
                                          pure turning and moving with correction.
        :param kp, ki, kd: PID gains for heading control.
        """
        self.linear_speed = linear_speed
        self.angular_speed_max = angular_speed_max
        self.waypoint_tolerance = waypoint_tolerance
        self.pure_turn_angle_threshold = pure_turn_angle_threshold

        self.heading_pid = PID(kp, ki, kd)
        
        # Proportional gain for steering while moving straight (used in mapping.py's simple control)
        # This will be replaced by PID in navigation.py, but kept for mapping.py's simpler logic
        self.k_linear_turn_correction = 0.5 

    def calculate_wheel_speeds_simple(self, robot_pose: dict, target_waypoint: list, dt: float) -> tuple[float, float]:
        """
        Calculates wheel speeds using a simple proportional control,
        suitable for mapping exploration (as in original mapping.py).
        This method does NOT use the PID controller.

        :param robot_pose: Dictionary with 'x', 'y', 'theta' (robot's current pose).
        :param target_waypoint: List [x, y] of the target waypoint.
        :param dt: Time step in seconds.
        :return: Tuple (left_speed, right_speed) for motors.
        """
        robot_x = robot_pose['x']
        robot_y = robot_pose['y']
        robot_theta = robot_pose['theta']

        dx = target_waypoint[0] - robot_x
        dy = target_waypoint[1] - robot_y
        angle_to_target = np.arctan2(dy, dx)

        angle_diff = angle_to_target - robot_theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) # Normalize to -pi to pi

        left_speed = 0.0
        right_speed = 0.0

        if abs(angle_diff) > 0.1: # If not aligned, turn
            if angle_diff > 0: # Turn left
                left_speed = -self.angular_speed_max * 0.5
                right_speed = self.angular_speed_max * 0.5
            else: # Turn right
                left_speed = self.angular_speed_max * 0.5
                right_speed = -self.angular_speed_max * 0.5
        else: # Aligned, move straight
            left_speed = self.linear_speed
            right_speed = self.linear_speed

        return left_speed, right_speed

    @staticmethod
    def _normalize_angle(angle):
        """Normalizes an angle to be between -PI and PI."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calculate_wheel_speeds_pid(self, robot_pose: dict, target_waypoint: list, dt: float) -> tuple[float, float]:
        """
        Calculates wheel speeds using a PID controller for heading,
        suitable for precise navigation (as in original navigation.py).

        :param robot_pose: Dictionary with 'x', 'y', 'theta' (robot's current pose).
        :param target_waypoint: List [x, y] of the target waypoint.
        :param dt: Time step in seconds.
        :return: Tuple (left_speed, right_speed) for motors.
        """
        robot_x = robot_pose['x']
        robot_y = robot_pose['y']
        robot_theta = robot_pose['theta']
        if robot_theta < 0:
            robot_theta += 2 * np.pi

        dx = target_waypoint[0] - robot_x
        dy = target_waypoint[1] - robot_y
        angle_to_target = np.arctan2(dy, dx)
        
        distance = np.sqrt(dx**2 + dy**2)

        angle_error = self._normalize_angle(angle_to_target - robot_theta)
        #angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error)) # Normalize to -pi to pi

        angular_velocity = self.heading_pid.update(angle_error, dt)
        angular_velocity = np.clip(angular_velocity, -self.angular_speed_max, self.angular_speed_max)

        left_speed = 0.0
        right_speed = 0.0

        if abs(angle_error) > self.pure_turn_angle_threshold:
            # Pure turning
            left_speed = -angular_velocity
            right_speed = angular_velocity
        else:
            # Move forward with angular correction
            # Reduce linear speed slightly if still correcting, or keep full linear speed
            base_speed = MAX_SPEED * 0.5
            self.linear_speed = base_speed * min(1.0, distance / self.waypoint_tolerance)
        
            forward_speed = self.linear_speed * (1 - abs(angle_error) / self.pure_turn_angle_threshold * 0.5) 
            forward_speed = max(0.1, forward_speed) # Ensure minimum forward speed

            left_speed = forward_speed - angular_velocity
            right_speed = forward_speed + angular_velocity
        
        return left_speed, right_speed

    def is_waypoint_reached(self, robot_pose: dict, target_waypoint: list) -> bool:
        """
        Checks if the robot has reached the target waypoint.

        :param robot_pose: Dictionary with 'x', 'y' (robot's current pose).
        :param target_waypoint: List [x, y] of the target waypoint.
        :return: True if the waypoint is reached, False otherwise.
        """
        distance = np.sqrt((target_waypoint[0] - robot_pose['x'])**2 + 
                            (target_waypoint[1] - robot_pose['y'])**2)
        return distance < self.waypoint_tolerance

    def reset_pid(self):
        """Resets the internal PID controller state."""
        self.heading_pid.reset()
        
        