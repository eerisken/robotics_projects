# behavior_tree.py
import py_trees
import py_trees.composites
import py_trees.behaviours # This import is generally for basic behaviours like Success, Failure, Running
import py_trees.decorators # Added import for decorators
import os

# Import custom modules
from blackboard import Blackboard
from mapping import MapEnvironment, MoveAroundTable
from planning import ComputePathToLowerLeftCorner, ComputePathToSink, ComputePathToGoal
from robot_state_updater import UpdateRobotPose # Import the new behavior
from navigation import NavigateToWaypoints
from frontal_obstacle_avoidance import FrontalObstacleAvoidance
from stuck_watchdog import StuckWatchdogWithMapFusion
from fold_arm import FoldArm
from pick_place2 import PickJar, PlaceJar
from turn_behaviour import TurnToAngle

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

class ReplanIfRequested(py_trees.behaviour.Behaviour):
    def __init__(self, name, blackboard):
        super(ReplanIfRequested, self).__init__(name)
        self.blackboard = blackboard
        self.name = name
        self.goal_name = self.blackboard.current_goal
  
    def update(self):
        if self.blackboard.replan_needed:
            self.logger.debug(f"{self.name}: [Replan] Triggered.")
        
            if self.goal_name == "goal_lower_left":
                ComputePathToLowerLeftCorner(blackboard)
            else:
                ComputePathToSink(blackboard)
    
            self.blackboard.replan_needed = False
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class SetCustomBlackboardVar(py_trees.behaviour.Behaviour):
    def __init__(self, name, blackboard, attr_name, value):
        super(SetCustomBlackboardVar, self).__init__(name)
        self.blackboard = blackboard
        self.attr_name = attr_name
        self.value = value

    def update(self):
        setattr(self.blackboard, self.attr_name, self.value)
        return py_trees.common.Status.SUCCESS

class CheckMissionCompleted(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot, blackboard):
        super(CheckMissionCompleted, self).__init__(name)
        self.blackboard = blackboard
        self.robot = robot

    def update(self):
        if getattr(self.blackboard, "mission_completed", True):
            # Stop robot
            self.robot.left_motor.setVelocity(0.0)
            self.robot.right_motor.setVelocity(0.0)
            return py_trees.common.Status.SUCCESS
        else:
            py_trees.common.Status.FAILURE
                    
def create_root_behaviour(robot, lidar, gps, compass, left_motor, right_motor, blackboard):
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

    # --- Sub-tree for navigating to a single goal with avoidance ---
    def create_goal_navigation_subtree(goal_name, robot, gps, compass, lidar, left_motor, right_motor, blackboard):
        #return compute_and_move_loop
        goal_navigation = py_trees.composites.Selector(
                            name=f"Avoid or Move to {goal_name}",
                            memory=False, # Always re-evaluate avoidance
                            children=[
                                FrontalObstacleAvoidance(f"Frontal Obstacle Avoidance", robot, lidar, left_motor, right_motor, blackboard),
                                NavigateToWaypoints(f"Move to {goal_name}", robot, gps, compass, left_motor, right_motor, blackboard)
                        ]
                    )
        
        return goal_navigation
        
    # 3. Navigation Tasks (after mapping or if map already exists)
    navigation_tasks = py_trees.composites.Sequence(
        name="Navigation Tasks",
        memory=True
    )
    
    set_done = SetCustomBlackboardVar(
        name="Set Mission Completed",
        blackboard=blackboard,           
        attr_name="mission_completed",
        value=True
    )

    lower_left_sequence = create_goal_navigation_subtree("Lower Left", robot, gps, compass, lidar, left_motor, right_motor, blackboard)
    sink_sequence = create_goal_navigation_subtree("Sink", robot, gps, compass, lidar, left_motor, right_motor, blackboard)
    
    navigation_tasks.add_children(children=[ComputePathToLowerLeftCorner(blackboard), lower_left_sequence,
                                            ComputePathToSink(blackboard), sink_sequence, set_done]

                                 ) # navigation_tasks

    # Root structure:
    # Selector: If map exists (Success), it will go to the next sibling (Navigation Tasks).
    # If map doesn't exist (Failure), it will try the next child (Map Creation Sequence).
    # Once the map is created, the Map Creation Sequence will succeed, and then the Root Sequence
    # will proceed to the Navigation Tasks.
    main_task_sequence.add_children([
        FoldArm("Fold Arm Behaviour", robot, blackboard),
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
    
    watchdog_and_replan = py_trees.composites.Sequence(name="Main Tasks", memory=False)
    stuck_watchdog = StuckWatchdogWithMapFusion(f"StuckWatchdog", robot, lidar, blackboard)
    replan = ReplanIfRequested(f"Replan If Requested", blackboard)
    watchdog_and_replan.add_children([stuck_watchdog, replan])
    
    root.add_child(py_trees.decorators.EternalGuard(
        name="Skip if Mission Done",
        condition=lambda: not blackboard.mission_completed,
        child=watchdog_and_replan
    ))

    new_root = py_trees.composites.Selector("MasterSelector", memory=True)
    new_root.add_children([
        CheckMissionCompleted("CheckMissionCompleted", robot, blackboard),  # as soon as bb.mission_done==True → succeed here
        root        # otherwise run your normal tree
    ])
          
    return new_root
    
def create_root_behaviour_pick_place(robot, lidar, gps, compass, left_motor, right_motor, blackboard):
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

    # --- Sub-tree for navigating to a single goal with avoidance ---
    def create_goal_navigation_subtree(goal_name, robot, gps, compass, lidar, left_motor, right_motor, blackboard):
        #return compute_and_move_loop
        goal_navigation = py_trees.composites.Selector(
                            name=f"Avoid or Move to {goal_name}",
                            memory=False, # Always re-evaluate avoidance
                            children=[
                                FrontalObstacleAvoidance(f"Frontal Obstacle Avoidance", robot, lidar, left_motor, right_motor, blackboard),
                                NavigateToWaypoints(f"Move to {goal_name}", robot, gps, compass, left_motor, right_motor, blackboard)
                        ]
                    )
        
        return goal_navigation
        
    # 3. Navigation Tasks (after mapping or if map already exists)
    pick_place_tasks = py_trees.composites.Sequence(
        name="Pick and Place Tasks",
        memory=True
    )
    
    set_done = SetCustomBlackboardVar(
        name="Set Mission Completed",
        blackboard=blackboard,           
        attr_name="mission_completed",
        value=True
    )

    turn_left = TurnToAngle("Turn Left 90°", robot, target_angle_deg=100)
    sink_sequence = create_goal_navigation_subtree("Sink", robot, gps, compass, lidar, left_motor, right_motor, blackboard)
    
    pick_place_tasks.add_children(children=[ComputePathToSink(blackboard), 
                                            sink_sequence, 
                                            PickJar("PickJar", robot, blackboard), 
                                            ComputePathToGoal("Compute path to table", blackboard, "goal_table"),
                                            create_goal_navigation_subtree("Move to table", robot, gps, compass, lidar, left_motor, right_motor, blackboard),
                                            PlaceJar("place jar", robot, blackboard),
                                            
                                            SetCustomBlackboardVar(
                                                name="Set Goal Sink",
                                                blackboard=blackboard,           
                                                attr_name="goal_sink",
                                                value=[0.44, -0.09]
                                            ),
                                            ComputePathToGoal("Compute path to Sink", blackboard, "goal_sink"),
                                            create_goal_navigation_subtree("Sink", robot, gps, compass, lidar, left_motor, right_motor, blackboard), 
                                            TurnToAngle("Turn Left 55°", robot, target_angle_deg=55),
                                            FoldArm("Fold Arm Behaviour", robot, blackboard),
                                            #TurnToAngle("Turn Left -20°", robot, target_angle_deg=-20),
                                            PickJar("PickJar", robot, blackboard),
                                            SetCustomBlackboardVar(
                                                name="Set Goal Table",
                                                blackboard=blackboard,           
                                                attr_name="goal_table",
                                                value=[0.03, -0.43]
                                            ),
                                            ComputePathToGoal("Compute path to table", blackboard, "goal_table"),
                                            create_goal_navigation_subtree("Move to table", robot, gps, compass, lidar, left_motor, right_motor, blackboard),
                                            PlaceJar("place jar", robot, blackboard),                                            
                                            
                                            #TurnToAngle("Turn Left 10°", robot, target_angle_deg=10),
                                            SetCustomBlackboardVar(
                                                name="Set Goal Sink",
                                                blackboard=blackboard,           
                                                attr_name="goal_sink",
                                                value=[0.43, -0.09]
                                            ),
                                            ComputePathToGoal("Compute path to Sink", blackboard, "goal_sink"), 
                                            create_goal_navigation_subtree("Sink", robot, gps, compass, lidar, left_motor, right_motor, blackboard), 
                                            #TurnToAngle("Turn Left 10°", robot, target_angle_deg=10),
                                            FoldArm("Fold Arm Behaviour", robot, blackboard),
                                            PickJar("PickJar", robot, blackboard), 
                                            SetCustomBlackboardVar(
                                                name="Set Goal Table",
                                                blackboard=blackboard,           
                                                attr_name="goal_table",
                                                value=[0.03, -0.4]
                                            ),
                                            ComputePathToGoal("Compute path to table", blackboard, "goal_table"),
                                            create_goal_navigation_subtree("Move to table", robot, gps, compass, lidar, left_motor, right_motor, blackboard),
                                            PlaceJar("place jar", robot, blackboard),
                                            set_done]

                                 ) 

    main_task_sequence.add_children([
        FoldArm("Fold Arm Behaviour", robot, blackboard),
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
        pick_place_tasks # This is a sibling to the Selector, executed after the Selector completes successfully          
    ])
    
    watchdog_and_replan = py_trees.composites.Sequence(name="Main Tasks", memory=False)
    stuck_watchdog = StuckWatchdogWithMapFusion(f"StuckWatchdog", robot, lidar, blackboard)
    replan = ReplanIfRequested(f"Replan If Requested", blackboard)
    watchdog_and_replan.add_children([stuck_watchdog, replan])
    
    """
    root.add_child(py_trees.decorators.EternalGuard(
        name="Skip if Mission Done",
        condition=lambda: not blackboard.mission_completed,
        child=watchdog_and_replan
    ))
    """
    
    new_root = py_trees.composites.Selector("MasterSelector", memory=True)
    new_root.add_children([
        CheckMissionCompleted("CheckMissionCompleted", robot, blackboard),  # as soon as bb.mission_done==True → succeed here
        root        # otherwise run your normal tree
    ])
          
    return new_root

