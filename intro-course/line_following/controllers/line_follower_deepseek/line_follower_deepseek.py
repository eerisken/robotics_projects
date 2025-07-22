from controller import Robot, DistanceSensor, Motor

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Constants
MAX_SPEED = 6.28
BASE_SPEED = 0.5 * MAX_SPEED
TURN_COEFF = 0.4  # Sharpness of corrective turns
SEARCH_SPEED = 0.3 * MAX_SPEED  # Right spin speed
THRESHOLD = 500  # Below this = detecting line

# Setup 3 ground sensors
gs = [robot.getDevice(name) for name in ['gs0', 'gs1', 'gs2']]
for sensor in gs:
    sensor.enable(timestep)

# Setup motors
left = robot.getDevice('left wheel motor')
right = robot.getDevice('right wheel motor')
left.setPosition(float('inf'))
right.setPosition(float('inf'))

while robot.step(timestep) != -1:
    # Read all 3 sensors
    l, m, r = [s.getValue() for s in gs]
    
    # Only consider line lost if ALL sensors see light
    if l > THRESHOLD and m > THRESHOLD and r > THRESHOLD:
        # Lost line - spin right
        left.setVelocity(SEARCH_SPEED)
        right.setVelocity(-SEARCH_SPEED)
    else:
        # Line following logic using all 3 sensors
        if m < THRESHOLD:  # Center on line
            left_speed = BASE_SPEED
            right_speed = BASE_SPEED
        elif l < THRESHOLD:  # Left on line
            left_speed = BASE_SPEED * TURN_COEFF
            right_speed = BASE_SPEED
        else:  # Right on line
            left_speed = BASE_SPEED
            right_speed = BASE_SPEED * TURN_COEFF
        
        left.setVelocity(left_speed)
        right.setVelocity(right_speed)