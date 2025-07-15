# webots_robot_controller.py
from controller import Supervisor, Robot, GPS, Compass, Lidar, Motor
import py_trees
import py_trees.console as console
import sys
import os
import time

# Add the directory containing your custom modules to the Python path
# This assumes the controller is in a subfolder, and the modules are in the same parent folder
# Or, if all files are in the same directory as the controller:
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

# Import custom modules
from blackboard import Blackboard
from behavior_tree import create_root_behaviour

# Set up logging for py_trees
#console.enable_colour(True)
py_trees.logging.level = py_trees.logging.Level.DEBUG

def run_robot_controller():
    """
    Initializes the Webots robot, sensors, motors, and runs the behaviour tree.
    """
    robot = Supervisor()
    
    # Get the basic time step of the current world
    TIME_STEP = int(robot.getBasicTimeStep())

    # Initialize sensors
    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    lidar = robot.getDevice("Hokuyo URG-04LX-UG01")    

    # Initialize motors (assuming Tiago's wheel motors)
    left_motor = robot.getDevice("wheel_left_joint")
    right_motor = robot.getDevice("wheel_right_joint")
    
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()
    gps.enable(TIME_STEP)
    compass.enable(TIME_STEP)

    # Set motor positions to infinity for velocity control
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Create the blackboard instance
    blackboard = Blackboard()

    # Create the root behaviour tree
    root = create_root_behaviour(robot, lidar, gps, compass, left_motor, right_motor, blackboard) 
    
    # Create the behaviour tree object
    tree = py_trees.trees.BehaviourTree(root)

    # Set up debugging and logging
    # py_trees.display.render_dot_tree(tree.root, name="Tiago_Behavior_Tree") # Requires graphviz
    # py_trees.logging.print_tree(tree.root) # Simple text tree

    # Tick the tree
    try:
        tree.setup(timeout=15) # Give some time for setup, e.g., sensor warm-up
    except py_trees.exceptions.InvalidNodeError as e:
        print(f"Tree setup failed: {e}")
        #robot.cleanup()
        return

    print("Behaviour tree setup complete. Starting simulation loop.")

    # Main simulation loop
    while robot.step(TIME_STEP) != -1:
        tree.tick()
        # You can add additional logging or checks here if needed
        # For example, print current robot pose
        # print(f"Robot Pose: X={blackboard.robot_pose['x']:.2f}, Y={blackboard.robot_pose['y']:.2f}, Theta={np.degrees(blackboard.robot_pose['theta']):.2f}°")

        # If the root behaviour has succeeded or failed, you might want to stop or reset
        if tree.root.status == py_trees.common.Status.SUCCESS:
            print("Robot has successfully completed all tasks! Stopping.")
            break
        elif tree.root.status == py_trees.common.Status.FAILURE:
            print("Robot behavior tree failed! Stopping.")
            break
            
    print("Simulation loop ended.")
    pass

run_robot_controller()

