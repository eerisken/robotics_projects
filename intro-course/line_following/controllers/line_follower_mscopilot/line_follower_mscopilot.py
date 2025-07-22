from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28

robot = Robot()

# Sensor setup
left_sensor = robot.getDevice('gs0')
center_sensor = robot.getDevice('gs1')
right_sensor = robot.getDevice('gs2')
for s in [left_sensor, center_sensor, right_sensor]:
    s.enable(TIME_STEP)

# Motor setup
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Control constants
Kp = 0.005  # Tune this value if turns are still too shallow

while robot.step(TIME_STEP) != -1:
    l = left_sensor.getValue()
    c = center_sensor.getValue()
    r = right_sensor.getValue()

    # Assume black = low value, white = high value
    # Compute weighted average "error" based on sensor values
    error = (r - l)  # simple bias: negative if line is on right, positive if on left
    correction = Kp * error

    left_speed = MAX_SPEED * (1.0 - correction)
    right_speed = MAX_SPEED * (1.0 + correction)

    # Clamp motor speed
    left_motor.setVelocity(max(min(left_speed, MAX_SPEED), -MAX_SPEED))
    right_motor.setVelocity(max(min(right_speed, MAX_SPEED), -MAX_SPEED))
