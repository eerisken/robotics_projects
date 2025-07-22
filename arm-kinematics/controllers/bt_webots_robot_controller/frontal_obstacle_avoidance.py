import py_trees
import time
import numpy as np

class FrontalObstacleAvoidance(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, lidar, left_motor, right_motor, blackboard, threshold=0.2):
        super(FrontalObstacleAvoidance, self).__init__(name)
        self.name = name
        self.blackboard = blackboard
        self.robot = robot
        self.lidar = lidar
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.threshold = threshold
        self.state = "IDLE"
        self.start_time = None
        self.avoid_direction = "left"  # or "right"

    def initialise(self):
        self.state = "IDLE"
        self.start_time = None

    def update(self):
        front_distances = np.array(self.lidar.getRangeImage())
        lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
        lidar_field_of_view = self.lidar.getFov()
        angles = np.linspace(lidar_field_of_view/2, -lidar_field_of_view/2, lidar_horizontal_resolution)
        
        obstacle_nearby = any(d < self.threshold for angle, d in zip(angles, front_distances) if -0.3 < angle < 0.3)

        if self.state == "IDLE":
            if obstacle_nearby:
                self.avoid_direction = "left" if self._left_side_clearer(zip(angles, front_distances)) else "right"
                self.start_time = time.time()
                self.state = "TURN_AWAY"
                self.logger.info(f"{self.name}: Obstacle detected, turning {self.avoid_direction}")
                return py_trees.common.Status.RUNNING
            return py_trees.common.Status.FAILURE  # no obstacle → let path following take over

        elif self.state == "TURN_AWAY":
            self._turn(self.avoid_direction)
            if time.time() - self.start_time > 0.5:
                self.start_time = time.time()
                self.state = "MOVE_FORWARD"
            return py_trees.common.Status.RUNNING

        elif self.state == "MOVE_FORWARD":
            self._move_forward()
            if time.time() - self.start_time > 1.0:
                self.start_time = time.time()
                self.state = "TURN_BACK"
            return py_trees.common.Status.RUNNING

        elif self.state == "TURN_BACK":
            self._turn("right" if self.avoid_direction == "left" else "left")
            if time.time() - self.start_time > 0.5:
                self.state = "IDLE"
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        
    def set_wheel_speeds(self, left_speed, right_speed):
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
    def _turn(self, direction):
        speed = 2.0
        if direction == "left":
            self.set_wheel_speeds(-speed, speed)
        else:
            self.set_wheel_speeds(speed, -speed)
        
    def _move_forward(self):
        self.set_wheel_speeds(3.0, 3.0)

    def _left_side_clearer(self, front_distances):
        left = [d for a, d in front_distances if 0.0 < a < 0.5]
        right = [d for a, d in front_distances if -0.5 < a < 0.0]
        return np.mean(left) > np.mean(right)
