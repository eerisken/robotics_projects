# obstacle_avoidance.py
import py_trees
import math
from blackboard import Blackboard
from obstacle_avoidance_behaviors import CheckFrontObstacle, TurnAwayFromObstacle, MoveClearOfObstacle # Import the new behaviors

class ObstacleAvoidance(py_trees.behaviour.Behaviour):
    """
    A behavior that encapsulates a subtree for obstacle avoidance.
    It checks for obstacles and, if found, executes a sequence of
    turning and moving to clear the path.
    """
    def __init__(self, name, robot, lidar, left_motor, right_motor, blackboard):
        super(ObstacleAvoidance, self).__init__(name)
        self.robot = robot
        self.blackboard = blackboard
        self.lidar = lidar
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.avoidance_tree = None # This will hold the internal avoidance subtree
        self.logger.debug("%s: Initialized ObstacleAvoidance (wrapper)" % self.name)

    def initialise(self):
        """
        Initialises the internal avoidance subtree.
        """
        # Create the internal avoidance subtree
        # Selector: If obstacle in front (CheckFrontObstacle returns SUCCESS), then execute avoidance sequence.
        # If no obstacle (CheckFrontObstacle returns FAILURE), the Avoidance Sequence fails.
        # Since it's the only child of the Avoidance Selector, the Avoidance Selector will also return FAILURE.
        # This FAILURE will allow the parent selector in tiago_controller.py to tick MoveToWaypoints.
        self.avoidance_tree = py_trees.composites.Selector(
            name="Avoidance Selector",
            memory=False, # Always re-evaluate obstacle presence
            children=[
                py_trees.composites.Sequence(
                    name="Avoidance Sequence",
                    memory=True, # Remember state of avoidance steps
                    children=[
                        CheckFrontObstacle("Check Front Obstacle", self.robot, self.lidar, self.blackboard),
                        TurnAwayFromObstacle("Turn Away From Obstacle", self.robot, self.lidar, self.left_motor, self.right_motor, self.blackboard),
                        MoveClearOfObstacle("Move Clear Of Obstacle", self.robot, self.lidar, self.left_motor, self.right_motor, self.blackboard)
                    ]
                )
                # Removed the py_trees.behaviours.Success fallback.
                # If CheckFrontObstacle fails, the Avoidance Sequence fails,
                # and the Avoidance Selector will correctly return FAILURE.
            ]
        )
        
        self.avoidance_tree.setup_with_descendants()
        tree = py_trees.trees.BehaviourTree(self.avoidance_tree)
        tree.setup(timeout=15)

        self.logger.debug("%s: Internal avoidance tree set up." % self.name)
        
        self.status = py_trees.common.Status.RUNNING 

    def update(self):
        """
        Ticks the internal avoidance behavior tree.
        """
        if self.avoidance_tree is None:
            self.logger.error("%s: Avoidance tree not initialised." % self.name)
            return py_trees.common.Status.FAILURE

        # --- Added try-except block here ---
        try:
            self.avoidance_tree.tick()
        except Exception as e:
            self.logger.error(f"{self.name}: An error occurred while ticking internal avoidance tree: {e}")
            self.feedback_message = f"Internal avoidance error: {e}"
            return py_trees.common.Status.FAILURE
        # --- End of try-except block ---

        # Explicitly check the status of the internal tree
        internal_tree_status = self.avoidance_tree.status
        self.feedback_message = self.avoidance_tree.feedback_message

        # If the internal tree somehow returns INVALID, convert it to FAILURE
        # This ensures the parent selector can proceed to MoveToWaypoints
        if internal_tree_status == py_trees.common.Status.INVALID:
            self.logger.warning(f"{self.name}: Internal avoidance tree returned INVALID status. Converting to FAILURE.")
            self.logger.warning(f"{self.name}: Internal avoidance tree returned feedback message: {self.feedback_message}")
            return py_trees.common.Status.FAILURE
        
        return internal_tree_status

    def terminate(self, new_status):
        """
        Terminates the internal avoidance subtree.
        """
        if self.avoidance_tree is not None:
            self.avoidance_tree.terminate(new_status)
        self.logger.debug("%s: Terminated with status %s" % (self.name, new_status))

