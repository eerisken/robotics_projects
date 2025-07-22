# obstacle_avoidance_behaviors.py
import py_trees
import math
from blackboard import Blackboard

# Constants for the robot and simulation
TIME_STEP = 32 # milliseconds
MAX_VELOCITY = 0.5 # m/s
WHEEL_RADIUS = 0.0975 # meters (approx for Tiago's wheels)
AXLE_LENGTH = 0.47 # meters (approx distance between wheels)
AVOIDANCE_TURN_VELOCITY = 0.2 # m/s
AVOIDANCE_FORWARD_DISTANCE = 0.5 # meters to move forward after turning

class ObstacleAvoidanceHelper:
    """
    A helper class to encapsulate common Lidar-related calculations.
    This can be inherited or used as a mixin if multiple behaviors need it.
    For now, we'll put the method directly into each class for clarity.
    """
    def _get_min_distance_in_sector(self, lidar_range_image, num_beams, fov, angle_start_rad, angle_end_rad):
        """
        Calculates the minimum distance in a specified angular sector of the Lidar's FOV.
        Angles are relative to the Lidar's center (0 is straight ahead).
        """
        # Convert angles from radians relative to FOV center to beam indices
        # Lidar beams are typically ordered from -fov/2 to +fov/2
        start_beam_idx = int(num_beams * (0.5 + (angle_start_rad / fov)))
        end_beam_idx = int(num_beams * (0.5 + (angle_end_rad / fov)))

        # Ensure indices are within valid range [0, num_beams-1]
        start_beam_idx = max(0, start_beam_idx)
        end_beam_idx = min(num_beams, end_beam_idx)
        
        # Ensure start_beam_idx is less than or equal to end_beam_idx
        if start_beam_idx > end_beam_idx:
            start_beam_idx, end_beam_idx = end_beam_idx, start_beam_idx

        # Extract the relevant slice of the lidar_range_image
        sector_ranges = lidar_range_image[start_beam_idx:end_beam_idx]

        # Return the minimum distance in the sector, or max range if no detections
        return min(sector_ranges or [self.lidar.getMaxRange()])
        
class CheckFrontObstacle(py_trees.behaviour.Behaviour, ObstacleAvoidanceHelper):
    """
    A condition behavior that checks if an obstacle is directly in front of the robot.
    Returns SUCCESS if an obstacle is detected, FAILURE otherwise.
    """
    def __init__(self, name, robot, lidar, blackboard):
        super(CheckFrontObstacle, self).__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.lidar = lidar
        self.lidar.enable(TIME_STEP)
        self.obstacle_distance_threshold = 0.6 # meters

        self.logger.debug("%s: Initialized CheckFrontObstacle" % self.name)

    def initialise(self):
        """
        Initializes the behavior. For a condition, it's usually ready to run.
        """
        self.status = py_trees.common.Status.RUNNING # Ensure status is not INVALID on first tick
        self.feedback_message = "Ready to check for obstacles."
        self.logger.debug("%s: Ready to check for obstacles" % self.name)
        
    def update(self):
        # Added this debug line to confirm if update is being called
        self.logger.debug("%s: In UPDATE METHOD of CheckFrontObstacle." % self.name)

        lidar_range_image = self.lidar.getRangeImage()
        if not lidar_range_image:
            self.feedback_message = "Lidar data not available."
            return py_trees.common.Status.FAILURE

        num_beams = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        
        # Define front sector angles (e.g., -15 to +15 degrees relative to Lidar center)
        angle_margin = math.radians(15) # 15 degrees in radians
        front_min_angle = -angle_margin
        front_max_angle = angle_margin

        min_front_dist = self._get_min_distance_in_sector(
            lidar_range_image, num_beams, fov, front_min_angle, front_max_angle
        )

        if min_front_dist < self.obstacle_distance_threshold:
            self.feedback_message = f"Obstacle detected at {min_front_dist:.2f}m."
            self.logger.debug("%s: In UPDATE METHOD of IF DIST CALCULATED." % self.name)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "No obstacle in front."
            self.logger.debug(f"{self.name}: In UPDATE METHOD of ELSE DIST CALCULATED. MIN FRON DIST = {min_front_dist}")
            return py_trees.common.Status.FAILURE
            
    def terminate(self, new_status):
        """
        Terminates the behavior. For a condition, usually no specific cleanup.
        """
        self.logger.debug("CheckFrontObstacles %s: Terminated with status %s" % (self.name, new_status))


class TurnAwayFromObstacle(py_trees.behaviour.Behaviour, ObstacleAvoidanceHelper):
    """
    Turns the robot towards the side with more free space.
    Returns RUNNING while turning, SUCCESS when turn is complete.
    """
    def __init__(self, name, robot, lidar, left_motor, right_motor, blackboard):
        super(TurnAwayFromObstacle, self).__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.lidar = lidar

        self.turn_duration_steps = 0
        self.current_step = 0
        self.feedback_message = "Initialized TurnAwayFromObstacle"
        self.logger.debug("%s: Initialized TurnAwayFromObstacle" % self.name)

    def initialise(self):
        self.current_step = 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        lidar_range_image = self.lidar.getRangeImage()
        if not lidar_range_image:
            self.feedback_message = "Lidar data not available for turning decision. Returning FAILURE."
            self.logger.warning("%s: Lidar data missing, cannot determine turn direction. Returning FAILURE." % self.name)
            self.status = py_trees.common.Status.FAILURE # Explicitly set status to FAILURE
            # return # Do NOT return here. Let the method complete.
        else:
            num_beams = self.lidar.getHorizontalResolution()
            fov = self.lidar.getFov()

            # Define left and right sector angles (e.g., 30-60 degrees from center)
            angle_inner = math.radians(15)
            angle_outer = math.radians(30)

            left_min_angle = -angle_outer
            left_max_angle = -angle_inner
            right_min_angle = angle_inner
            right_max_angle = angle_outer

            min_left_dist = self._get_min_distance_in_sector(
                lidar_range_image, num_beams, fov, left_min_angle, left_max_angle
            )
            min_right_dist = self._get_min_distance_in_sector(
                lidar_range_image, num_beams, fov, right_min_angle, right_max_angle
            )

            turn_angle = math.pi / 4 # Turn 45 degrees
            angular_vel = (AVOIDANCE_TURN_VELOCITY - (-AVOIDANCE_TURN_VELOCITY)) / AXLE_LENGTH
            self.turn_duration_steps = int((turn_angle / angular_vel) * (1000 / TIME_STEP))

            if min_left_dist > min_right_dist:
                self.turn_direction = "LEFT"
                self.logger.info("%s: Determined to turn LEFT." % self.name)
            else:
                self.turn_direction = "RIGHT"
                self.logger.info("%s: Determined to turn RIGHT." % self.name)
            self.feedback_message = f"Starting turn {self.turn_direction}."
            self.status = py_trees.common.Status.RUNNING # Set initial status to RUNNING
        
        self.logger.debug("%s: Initialised" % self.name)

    def update(self):
        # If initialise failed, we should propagate that failure.
        if self.status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        if self.current_step < self.turn_duration_steps:
            if self.turn_direction == "LEFT":
                self.left_motor.setVelocity(-AVOIDANCE_TURN_VELOCITY)
                self.right_motor.setVelocity(AVOIDANCE_TURN_VELOCITY)
            else: # RIGHT
                self.left_motor.setVelocity(AVOIDANCE_TURN_VELOCITY)
                self.right_motor.setVelocity(-AVOIDANCE_TURN_VELOCITY)
            self.current_step += 1
            self.feedback_message = f"Turning {self.turn_direction} to avoid obstacle."
            return py_trees.common.Status.RUNNING
        else:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            self.feedback_message = f"Finished turning {self.turn_direction}."
            self.logger.info("%s: Turn complete." % self.name)
            return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug("TurnAwayFromObstacle %s: Terminated with status %s" % (self.name, new_status))

class MoveClearOfObstacle(py_trees.behaviour.Behaviour, ObstacleAvoidanceHelper):
    """
    Moves the robot forward a fixed distance to clear the obstacle.
    Returns RUNNING while moving, SUCCESS when distance is covered and path is clear.
    """
    def __init__(self, name, robot, lidar, left_motor, right_motor, blackboard):
        super(MoveClearOfObstacle, self).__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.lidar = lidar

        self.forward_duration_steps = 0
        self.current_step = 0
        self.clear_distance_threshold = 1.2 # meters
        self.feedback_message = "Initialized MoveClearOfObstacle"
        self.logger.debug("%s: Initialized MoveClearOfObstacle" % self.name)

    def initialise(self):
        self.current_step = 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.forward_duration_steps = int((AVOIDANCE_FORWARD_DISTANCE / (MAX_VELOCITY / 2)) * (1000 / TIME_STEP))
        self.feedback_message = "Starting to move forward to clear."
        self.logger.info("%s: Moving forward for %d steps to clear obstacle." % (self.name, self.forward_duration_steps))
        self.status = py_trees.common.Status.RUNNING # Set initial status to RUNNING

    def update(self):
        # If initialise failed, we should propagate that failure.
        if self.status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE
            
        if self.current_step < self.forward_duration_steps:
            self.left_motor.setVelocity(MAX_VELOCITY / 2)
            self.right_motor.setVelocity(MAX_VELOCITY / 2)
            self.current_step += 1
            self.feedback_message = "Moving forward to clear obstacle."
            return py_trees.common.Status.RUNNING
        else:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            # Re-check if obstacle is truly cleared
            lidar_range_image = self.lidar.getRangeImage()
            if lidar_range_image:
                num_beams = self.lidar.getHorizontalResolution()
                fov = self.lidar.getFov()

                # Define front sector angles
                angle_margin = math.radians(15)
                front_min_angle = -angle_margin
                front_max_angle = angle_margin

                min_front_dist = self._get_min_distance_in_sector(
                    lidar_range_image, num_beams, fov, front_min_angle, front_max_angle
                )

                if min_front_dist > self.clear_distance_threshold:
                    self.feedback_message = "Obstacle cleared. Path is free."
                    self.logger.info("%s: Obstacle cleared. Returning SUCCESS." % self.name)
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = "Obstacle still present. Re-evaluating."
                    self.logger.info("%s: Obstacle still present after move. Returning FAILURE to re-trigger avoidance." % self.name)
                    # If still blocked, return FAILURE to allow the selector to re-evaluate avoidance.
                    return py_trees.common.Status.FAILURE
            else:
                self.feedback_message = "Lidar data not available for clearance check."
                self.logger.warning("%s: Lidar data missing for clearance check. Assuming not clear." % self.name)
                return py_trees.common.Status.FAILURE # Cannot confirm clear, so fail to re-evaluate

    def terminate(self, new_status):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.logger.debug("MoveClearOfObstacle %s: Terminated with status %s" % (self.name, new_status))

