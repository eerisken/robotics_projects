from controller import Robot

robot = Robot()
TIME_STEP = 32
MAX_SPEED = 6.28

# Distance sensor init
ps = []
for i in range(8):
    ps.append(robot.getDevice('ps'+str(i)))
    ps[i].enable(TIME_STEP)

# Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)

# Encoders
leftEncoder = leftMotor.getPositionSensor()
rightEncoder = rightMotor.getPositionSensor()
leftEncoder.enable(TIME_STEP)
rightEncoder.enable(TIME_STEP)

DIST_THRESHOLD = 380
state = "DRIVING_FORWARD"

# helper
def set_speed(l, r):
    leftMotor.setVelocity(l * MAX_SPEED)
    rightMotor.setVelocity(r * MAX_SPEED)

# Robot specs
WHEEL_RADIUS = 0.0205   # meters
WHEEL_DISTANCE = 0.053  # meters
CALIBRATION_FACTOR = 1.1
TARGET_WHEEL_ROTATION = CALIBRATION_FACTOR * (3.1416 * (WHEEL_DISTANCE / 2)) / WHEEL_RADIUS

init_left = 0
init_right = 0
front_left_before = 0
front_right_before = 0
TOLERANCE = 80   # Sensor match tolerance

while robot.step(TIME_STEP) != -1:
    ps_val = [psv.getValue() for psv in ps]
    front_left = ps_val[7]
    front_right = ps_val[0]
    left_side = ps_val[5]

    #print(ps_val)

    if state == "DRIVING_FORWARD":
        set_speed(0.5, 0.5)
        if front_left > DIST_THRESHOLD or front_right > DIST_THRESHOLD:
            state = "TURNING_AROUND"
            init_left = leftEncoder.getValue()
            init_right = rightEncoder.getValue()
            # Save the front sensor baseline
            front_left_before = front_left
            front_right_before = front_right
            
            print("obstacle_detected...")

    elif state == "TURNING_AROUND":
        set_speed(0.5, -0.5)
        left_traveled = leftEncoder.getValue() - init_left
        right_traveled = init_right - rightEncoder.getValue()

        
        if left_traveled >= TARGET_WHEEL_ROTATION and right_traveled >= TARGET_WHEEL_ROTATION:
           
            back_left = ps_val[4]
            back_right = ps_val[3]

            if abs(front_left_before - back_left) < TOLERANCE and abs(front_right_before - back_right) < TOLERANCE:
                set_speed(0, 0)
                print("turning...")
                state = "DRIVING_FORWARD_AGAIN"

    elif state == "DRIVING_FORWARD_AGAIN":
        set_speed(0.5, 0.5)
        if front_left > DIST_THRESHOLD or front_right > DIST_THRESHOLD:
            print("obstacle_detected")
            state = "ROTATING_TO_WALL"

    elif state == "ROTATING_TO_WALL":
        
        if 'init_left_90' not in globals():
            init_left_90 = leftEncoder.getValue()
            init_right_90 = rightEncoder.getValue()
    
        set_speed(0.5, -0.5)
    
        left_traveled =  leftEncoder.getValue() - init_left_90   
        right_traveled = init_right_90 - rightEncoder.getValue()    
    

        TARGET_90_ROTATION = (1.5708 * (WHEEL_DISTANCE/2)) / WHEEL_RADIUS * 1.2
    
        if left_traveled >= TARGET_90_ROTATION and right_traveled >= TARGET_90_ROTATION:
            set_speed(0, 0)
            print("turning...")
            del init_left_90, init_right_90
            state = "FOLLOW_WALL_FORWARD"

    elif state == "FOLLOW_WALL_FORWARD":

        if left_side > DIST_THRESHOLD:
            set_speed(0.5, 0.5)
            
        else:
            set_speed(0, 0)  
            print("Stopping...")