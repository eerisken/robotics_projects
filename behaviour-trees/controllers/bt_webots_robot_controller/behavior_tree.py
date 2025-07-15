# behavior_tree.py
import py_trees
import py_trees.composites
import py_trees.behaviours # This import is generally for basic behaviours like Success, Failure, Running
import py_trees.decorators # Added import for decorators
import os

# Import custom modules
from blackboard import Blackboard
from mapping import MapEnvironment, MoveAroundTable
from planning import ComputePathToLowerLeftCorner, ComputePathToSink
from robot_state_updater import UpdateRobotPose # Import the new behavior
from navigation import NavigateToWaypoints

class CheckMapExists(py_trees.behaviour.Behaviour):
    """
    A custom behaviour to check if the map and c-space files exist on disk.
    """
    def __init__(self, name, blackboard):
        super(CheckMapExists, self).__init__(name)
        self.blackboard = blackboard

    def initialise(self):
        self.blackboard.load_map() # Attempt to load map and set map_exists flag
        self.logger.debug(f"{self.name}: Initialized. Map exists: {self.blackboard.map_exists}")

    def update(self):
        if self.blackboard.map_exists:
            self.logger.info(f"{self.name}: Map and C-space files found. Success!")
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.info(f"{self.name}: No map or C-space files found. Failure!")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(f"{self.name}: Terminated with status {new_status}.")


def create_root_behaviour(robot, lidar, gps, compass, left_motor, right_motor, blackboard):
    """
    Creates and returns the root behaviour for the robot's behaviour tree.
    """
    """
    # Define the main sequence of operations
    root = py_trees.composites.Sequence(
        name="Robot Controller",
        memory=True # Remember the last running child
    )
    """
    
    # Root node: A Parallel node to run sensor updates continuously alongside tasks
    root = py_trees.composites.Parallel(
        name="Robot Controller",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(False) # All children must succeed, but it keeps ticking
    )

    # Always-on behaviour: Update Robot Pose
    update_pose = UpdateRobotPose(name="Update Robot Pose", robot=robot, gps=gps, compass=compass, blackboard=blackboard)
    root.add_child(update_pose)
    
    # Main Task Sequence (runs after pose update)
    main_task_sequence = py_trees.composites.Sequence(name="Main Tasks", memory=True)
    root.add_child(main_task_sequence)

    # 1. Check if map and c-space exist
    # This instance is for the Selector's direct check
    check_map_exists_main = CheckMapExists("Check Map Exists (Main)", blackboard)

    # This instance is specifically for the Inverter, to avoid shared parent error
    check_map_exists_for_inverter = CheckMapExists("Check Map Exists (Inverted)", blackboard)


    # 2. Mapping and C-space Generation (if map doesn't exist)
    # This parallel node succeeds if 'Move Around Table' succeeds.
    # 'Map the Environment' will continue running until then.
    mapping_and_cspace = py_trees.composites.Parallel(
        name="Mapping",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    map_environment = MapEnvironment(
        "Map the Environment",
        robot, lidar, gps, compass, blackboard
    )
    move_around_table = MoveAroundTable(
        "Move Around Table",
        robot, gps, compass, left_motor, right_motor, blackboard
    )
    mapping_and_cspace.add_children([map_environment, move_around_table])

    # 3. Navigation Tasks (after mapping or if map already exists)
    navigation_tasks = py_trees.composites.Sequence(
        name="Navigation Tasks",
        memory=True
    )
    
    # Task 1: Navigate to Lower Left Corner
    compute_path_lower_left = ComputePathToLowerLeftCorner(blackboard)
    navigate_lower_left = NavigateToWaypoints(
        "Navigate to Lower Left Corner",
        robot, gps, compass, left_motor, right_motor, blackboard 
    )
    lower_left_sequence = py_trees.composites.Sequence(name="Go to Lower Left", memory=True)
    lower_left_sequence.add_children([compute_path_lower_left, navigate_lower_left])

    # Task 2: Navigate to Sink
    compute_path_sink = ComputePathToSink(blackboard)
    navigate_sink = NavigateToWaypoints(
        "Navigate to Sink",
        robot, gps, compass, left_motor, right_motor, blackboard 
    )
    sink_sequence = py_trees.composites.Sequence(name="Go to Sink", memory=True)
    sink_sequence.add_children([compute_path_sink, navigate_sink])

    navigation_tasks.add_children([lower_left_sequence, sink_sequence]) 

    # Root structure:
    # Selector: If map exists (Success), it will go to the next sibling (Navigation Tasks).
    # If map doesn't exist (Failure), it will try the next child (Map Creation Sequence).
    # Once the map is created, the Map Creation Sequence will succeed, and then the Root Sequence
    # will proceed to the Navigation Tasks.
    
    main_task_sequence.add_children([
        py_trees.composites.Selector(
            name="Map Check or Create",
            memory=True,
            children=[
                check_map_exists_main, # This instance is for the Selector
                py_trees.composites.Sequence( # This sequence runs if check_map_exists_main fails
                    name="Map Creation Sequence",
                    memory=True,
                    children=[
                        py_trees.decorators.Inverter(name="Invert Map Check", child=check_map_exists_for_inverter), # Use the new instance here
                        mapping_and_cspace # Perform mapping
                    ]
                )
            ]
        ),
        navigation_tasks # This is a sibling to the Selector, executed after the Selector completes successfully
    ])

    return root

