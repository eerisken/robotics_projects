from controller import Supervisor, Robot, GPS, Lidar, Compass, Motor, Display
import numpy as np
import math
import sys
import os
import py_trees

from scipy import signal
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# --- Global Constants (moved into Probabilistic_Occupancy_Map or passed to it) ---
TIME_STEP = 64  # milliseconds
MAX_SPEED = 6.28  # rad/s (Taigolite's max speed)

# Waypoints for trajectory following (X, Y coordinates in Webots world's horizontal plane)
WAYPOINTS = [
            (-1.05, 0.24), (-1.68, -0.28), 
             (-1.65, -1.49), (-1.65, -3), (0.35, -2.8), 
             (0.35, -1.45), (0.38, -0.24), (-0.5, 0.7)  
            ]
WAYPOINTS = WAYPOINTS + list(reversed(WAYPOINTS))
WAYPOINT_THRESHOLD = 0.35  # meters, distance to consider waypoint reached

TIME_LIMIT_MS = 2 * 60 * 1000  # 2 minutes in milliseconds

# --- PID Controller Class ---
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

## Probabilistic Occupancy Map Class
## This class encapsulates all logic related to creating, updating, and manipulating the probabilistic occupancy grid.

class ProbabilisticOccupancyMap(py_trees.behaviour.Behaviour):
    # Mapping parameters
    MAP_RESOLUTION = 0.05  # meters per grid cell (5 cm)
    MAP_ORIGIN_X = -3  # -1.68 - 0.2
    MAP_ORIGIN_Y = -3.8   # -3.00 - 0.2
    MAP_X_DIM_METERS = 5.6
    MAP_Y_DIM_METERS = 6.4

    GRID_WIDTH = int(np.ceil(MAP_X_DIM_METERS / MAP_RESOLUTION))
    GRID_HEIGHT = int(np.ceil(MAP_Y_DIM_METERS / MAP_RESOLUTION)) # GRID_HEIGHT now refers to Y-dimension

    # Log-odds update values
    LOG_ODDS_OCCUPIED = np.log(0.9 / 0.1)
    LOG_ODDS_FREE = np.log(0.1 / 0.9) / 10.0
    LOG_ODDS_CLAMP_MAX = 5.0
    LOG_ODDS_CLAMP_MIN = -5.0

    # C-Space parameters
    C_SPACE_THRESHOLD_PROB = 0.6
    C_SPACE_THRESHOLD_LOG_ODDS = np.log(C_SPACE_THRESHOLD_PROB / (1 - C_SPACE_THRESHOLD_PROB))

    # Output directory for matplotlib figures
    OUTPUT_DIR = "map_outputs"

    def __init__(self, robot_radius, blackboard):
        super(ProbabilisticOccupancyMap, self).__init__(name)
        self.blackboard = blackboard
        self.blackboard.current_map = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.map_data = self.blackboard.current_map #np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.robot_radius = robot_radius # Robot radius for obstacle growing
        self._create_output_dir()
        print(f"Probabilistic_Occupancy_Map initialized. Map: {self.GRID_WIDTH}x{self.GRID_HEIGHT} cells. Output: '{self.OUTPUT_DIR}'")

    def _create_output_dir(self):
        """Creates the directory for saving matplotlib output images."""
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            print(f"Created output directory: {self.OUTPUT_DIR}")

    @staticmethod
    def _world_to_grid(x_world, y_world):
        """Converts world coordinates (x, y) to grid coordinates (col, row)."""
        col = int((x_world - ProbabilisticOccupancyMap.MAP_ORIGIN_X) / ProbabilisticOccupancyMap.MAP_RESOLUTION)
        row = int((y_world - ProbabilisticOccupancyMap.MAP_ORIGIN_Y) / ProbabilisticOccupancyMap.MAP_RESOLUTION)
        
        col = max(0, min(col, ProbabilisticOccupancyMap.GRID_WIDTH - 1))
        row = max(0, min(row, ProbabilisticOccupancyMap.GRID_HEIGHT - 1))
    
        return col, row

    @staticmethod
    def _grid_to_world(col, row):
        """Converts grid coordinates (col, row) to world coordinates (x, y) of the cell center."""
        x_world = col * ProbabilisticOccupancyMap.MAP_RESOLUTION + ProbabilisticOccupancyMap.MAP_ORIGIN_X + ProbabilisticOccupancyMap.MAP_RESOLUTION / 2
        y_world = row * ProbabilisticOccupancyMap.MAP_RESOLUTION + ProbabilisticOccupancyMap.MAP_ORIGIN_Y + ProbabilisticOccupancyMap.MAP_RESOLUTION / 2
        return x_world, y_world

    @staticmethod
    def _clamp(value, min_val, max_val):
        """Clamps a value between a min and max."""
        return max(min_val, min(value, max_val))

    @staticmethod
    def _clamp_world_coordinate(coord, map_origin, map_dim_meters):
        """Clamps a world coordinate to be within the defined map boundaries."""
        min_coord = map_origin
        max_coord = map_origin + map_dim_meters
        return max(min_coord, min(coord, max_coord - 1e-6)) # Subtract epsilon to stay within bounds

    @staticmethod
    def _log_odds_to_probability(log_odds):
        """Converts log-odds to probability."""
        return 1 - (1 / (1 + np.exp(log_odds)))

    @staticmethod
    def _bresenham_line(x0, y0, x1, y1):
        """Yields (x, y) coordinates for a line using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def update_map(self, robot_x, robot_y, robot_heading, lidar_range_image, lidar_horizontal_resolution, lidar_field_of_view, lidar_max_range):
        """
        Updates the probabilistic occupancy map based on Lidar readings.
        """
        robot_grid_col, robot_grid_row = self._world_to_grid(robot_x, robot_y)
        
        #print(f"Scan ranges: min={min(lidar_range_image[80:-80])}, max={max(lidar_range_image[80:-80])}")

        # Define the number of readings to exclude from each end
        EXCLUDE_READINGS = 80
        angles = np.linspace(lidar_field_of_view/2, -lidar_field_of_view/2, lidar_horizontal_resolution) #[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        ranges = lidar_range_image #[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        
        # Adjust the range of the loop to skip the first and last EXCLUDE_READINGS
        # The loop will now iterate from EXCLUDE_READINGS up to (lidar_horizontal_resolution - EXCLUDE_READINGS - 1)
        for i in range(EXCLUDE_READINGS, lidar_horizontal_resolution - EXCLUDE_READINGS):
            ray_angle_relative_to_lidar = angles[i] #-lidar_field_of_view / 2 + i * (lidar_field_of_view / (lidar_horizontal_resolution - 1))
            absolute_ray_angle = self._normalize_angle(robot_heading + ray_angle_relative_to_lidar)
            #absolute_ray_angle = robot_heading + ray_angle_relative_to_lidar 

            distance = ranges[i]
            
            effective_distance = distance
            if math.isinf(distance) or distance > lidar_max_range:
                effective_distance = lidar_max_range
            effective_distance = max(0.01, effective_distance) # Ensure positive and non-zero distance

            # Calculate the potential hit point in world coordinates
            hit_x_unclamped = robot_x + effective_distance * math.cos(absolute_ray_angle)
            hit_y_unclamped = robot_y + effective_distance * math.sin(absolute_ray_angle)

            # Clamp hit point to map's extent
            hit_x = self._clamp_world_coordinate(hit_x_unclamped, self.MAP_ORIGIN_X, self.MAP_X_DIM_METERS)
            hit_y = self._clamp_world_coordinate(hit_y_unclamped, self.MAP_ORIGIN_Y, self.MAP_Y_DIM_METERS)
            
            hit_grid_col, hit_grid_row = self._world_to_grid(hit_x, hit_y)

            # Ensure hit point is within grid boundaries for Bresenham
            hit_grid_row = self._clamp(hit_grid_row, 0, self.GRID_HEIGHT - 1)
            hit_grid_col = self._clamp(hit_grid_col, 0, self.GRID_WIDTH - 1)
            
            #self.map_data[hit_grid_row, hit_grid_col] = 5
            
            # Generate cells along the ray from robot to hit point (or max_range point)
            ray_cells = self._bresenham_line(robot_grid_col, robot_grid_row, hit_grid_col, hit_grid_row)
            #print("grid col row", robot_grid_col, robot_grid_row, hit_grid_col, hit_grid_row, len(ray_cells))
            
            # --- Unified Free and Occupied Space Marking Logic ---
            for col, row in ray_cells:
                if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
                    is_hit_cell = (col == hit_grid_col and row == hit_grid_row)
                    
                    if is_hit_cell and not math.isinf(distance) and distance <= lidar_max_range:
                        self.map_data[row, col] += self.LOG_ODDS_OCCUPIED
                    else:
                        self.map_data[row, col] += self.LOG_ODDS_FREE
                        
                    self.map_data[row, col] = self._clamp(
                        self.map_data[row, col],
                        self.LOG_ODDS_CLAMP_MIN,
                        self.LOG_ODDS_CLAMP_MAX
                    )
            
    def update_map2(self, robot_x, robot_y, robot_heading, lidar_range_image, lidar_horizontal_resolution, lidar_field_of_view, lidar_max_range):
        """
        Updates the probabilistic occupancy map based on Lidar readings.
        """
        robot_grid_col, robot_grid_row = self._world_to_grid(robot_x, robot_y)
        
        #print(f"Scan ranges: min={min(lidar_range_image[80:-80])}, max={max(lidar_range_image[80:-80])}")

        # Define the number of readings to exclude from each end
        EXCLUDE_READINGS = 80

        w_T_r = np.array([[np.cos(robot_heading), -np.sin(robot_heading), robot_x], 
                 [np.sin(robot_heading), np.cos(robot_heading), robot_y],
                 [0,0,1]])
                 
        angles = np.linspace(lidar_field_of_view/2, -lidar_field_of_view/2, lidar_horizontal_resolution)[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        ranges = np.minimum(lidar_range_image, 10.0)[EXCLUDE_READINGS:-EXCLUDE_READINGS]
        
        x_i = np.array([ranges*np.cos(angles), ranges*np.sin(angles), np.ones(ranges.shape[0])])
        D = w_T_r @ x_i
        
        for k in range(D.shape[1]):
            lx, ly = D[0, k], D[1, k]
            px_l, py_l = self._world_to_grid(lx, ly)
            # Check if pixel inside display bounds (usually 300x300 here)
            if 0 <= px_l < self.GRID_HEIGHT and 0 <= py_l < self.GRID_WIDTH:
                self.map_data[py_l, px_l] = min(self.map_data[py_l, px_l] + 0.01, 1.0)
                #v = int(self.map_data[py_l, px_l] * 255)
                #color = (v * 256**2 + v * 256 + v)
                #self.display.setColor(color)
                #self.display.drawPixel(px_l, py_l)    
            
    def grow_obstacles(self):
        """
        Grows obstacles in the probabilistic map to create a C-Space.
        Uses convolution with a circular kernel.
        Returns the grown probabilistic map.
        """
        binary_obstacles = (self.map_data >= self.C_SPACE_THRESHOLD_LOG_ODDS).astype(np.float32)

        growth_radius_pixels = int(np.ceil(self.robot_radius / self.MAP_RESOLUTION))
        kernel_size = 2 * growth_radius_pixels + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        center = growth_radius_pixels
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                if np.sqrt((i - center)**2 + (j - center)**2) <= growth_radius_pixels:
                    kernel[i, j] = 1.0
        
        convolved_map = signal.convolve2d(binary_obstacles, kernel, mode='same')

        max_conv_val = kernel.sum()
        if max_conv_val == 0: max_conv_val = 1

        normalized_convolved = np.clip(convolved_map / max_conv_val, 0.0, 1.0)
        grown_prob_map_log_odds = np.log(normalized_convolved / (1 - normalized_convolved + 1e-9) + 1e-9)
        
        final_grown_map = np.maximum(self.map_data, grown_prob_map_log_odds)
        final_grown_map = np.maximum(self.LOG_ODDS_CLAMP_MIN, np.minimum(final_grown_map, self.LOG_ODDS_CLAMP_MAX))
        
        return final_grown_map

    def compute_c_space(self, grown_map):
        """
        Computes the binary C-Space from the grown probabilistic map.
        """
        c_space_mask = grown_map >= self.C_SPACE_THRESHOLD_LOG_ODDS
        c_space = c_space_mask.astype(int)
        return c_space

    def get_map_data(self):
        """Returns the current probabilistic occupancy map data."""
        return self.map_data
        
    @staticmethod
    def _normalize_angle(angle):
        """Normalizes an angle to be between -PI and PI."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def save_map_as_image(self, map_data, title, filename_prefix, cmap='gray', vmin=None, vmax=None):
        """Saves the given map data as a matplotlib image."""
        fig, ax = plt.subplots(figsize=(self.GRID_WIDTH/10.0, self.GRID_HEIGHT/10.0), dpi=100)
        
        if vmin is None and vmax is None:
            if map_data.dtype == np.float32: # Probabilistic map (log-odds)
                display_data = self._log_odds_to_probability(map_data)
                vmin_val, vmax_val = 0.0, 1.0
            else: # Binary map (C-Space)
                display_data = map_data
                vmin_val, vmax_val = 0, 1
        else:
            display_data = map_data
            vmin_val, vmax_val = vmin, vmax

        # Transpose `display_data` to correctly align `(col, row)` with `imshow(x, y)`
        # imshow(data) where data[0,0] is top-left.
        # Our map's (0,0) is bottom-left, so we transpose and set origin='lower'.
        im = ax.imshow(display_data.T, cmap=cmap, origin='lower', vmin=vmin_val, vmax=vmax_val, interpolation='nearest')
        
        ax.set_title(title)
        ax.axis('off')

        if map_data.dtype == np.float32:
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Occupancy Probability')
        
        full_path = os.path.join(self.OUTPUT_DIR, f"{filename_prefix}.png")
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved '{title}' to {full_path}")

    def save_probabilistic_map_with_robot_path(self, robot_path, filename="probabilistic_map_with_robot_path.png"):
        """
        Saves the current probabilistic map as an RGB image with the robot's path overlaid.
        The robot_path should be a list of (world_x, world_y) tuples.
        """
        # Convert log-odds map to 0-1 probability map
        probabilities_map = self._log_odds_to_probability(self.map_data)
    
        # Initialize an RGB image array
        map_image_rgb = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH, 3), dtype=np.uint8)
    
        # Populate the map_image_rgb with grayscale values for the map itself
        for col in range(self.GRID_WIDTH):
            for row in range(self.GRID_HEIGHT):
                probability = probabilities_map[col, row]
                gray_value = int(probability * 255)
                # Apply Y-axis flip for image indexing
                map_image_rgb[self.GRID_HEIGHT - 1 - row, col, :] = gray_value
    
        # Draw the robot's path in red on the RGB image
        red_color_rgb = [255, 0, 0] # Pure Red
    
        for world_x, world_y in robot_path:
            grid_col, grid_row = self._world_to_grid(world_x, world_y)
    
            # Ensure path points are within map boundaries
            if 0 <= grid_col < self.GRID_WIDTH and 0 <= grid_row < self.GRID_HEIGHT:
                # Apply Y-axis flip for image indexing
                draw_row = self.GRID_HEIGHT - 1 - grid_row
                draw_col = grid_col
        
                # Draw a small 3x3 square for a thicker path (more visible)
                for r_offset in range(-1, 2): # -1, 0, 1
                    for c_offset in range(-1, 2): # -1, 0, 1
                        pr, pc = draw_row + r_offset, draw_col + c_offset
                        if 0 <= pr < self.GRID_HEIGHT and 0 <= pc < self.GRID_WIDTH:
                            map_image_rgb[pr, pc, :] = red_color_rgb
    
        # Now plot and save using matplotlib
        plt.figure(figsize=(8, 8)) # Adjust figure size as needed
        plt.imshow(map_image_rgb)
        plt.title('Final Probabilistic Map with Robot Path')
        plt.axis('off') # Turn off axes for a cleaner map
        plt.tight_layout()
        full_path = os.path.join(self.OUTPUT_DIR, f"{filename}.png")
        plt.savefig(full_path)
        plt.close() # Close the plot to free memory
 
class TaigoliteController:
    def __init__(self, robot_radius=0.2):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.gps = None
        self.compass = None
        self.left_motor = None
        self.right_motor = None
        self.lidar = None
        self.display = None
        #self.map_display = None

        self.heading_pid = PID(kp=2.5, ki=0.00, kd=0.1)

        # Instantiate the probabilistic occupancy map
        self.occupancy_map = ProbabilisticOccupancyMap(robot_radius=robot_radius)

        self.current_waypoint_index = 0
        self.trajectory_finished = False
        self.start_time = 0
        self.map_processed_and_saved = False
        
        self.path = []
        self.path_interval = 10 # Record path every 10 timesteps (adjust as needed)
        self.timestep_counter = 0

        self._init_devices()
        self.start_time = self.robot.getTime() * 1000
        
        print(f"TaigoliteController initialized.")

    def _init_devices(self):
        """Initializes all Webots robot devices."""
        self.gps = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice("compass")
        self.left_motor = self.robot.getDevice("wheel_left_joint")
        self.right_motor = self.robot.getDevice("wheel_right_joint")
        self.lidar = self.robot.getDevice("Hokuyo URG-04LX-UG01") # Or your Lidar's actual name
        self.display = self.robot.getDevice("display")
        #self.map_display = self.robot.getDevice("map_display")
        
        self.marker = self.robot.getFromDef("marker").getField("translation") # z = 0.0131382

        if self.gps:
            self.gps.enable(self.time_step)
        else:
            print("WARNING: GPS device 'gps' not found.", file=sys.stderr)

        if self.compass:
            self.compass.enable(self.time_step)
        else:
            print("WARNING: Compass device 'compass' not found.", file=sys.stderr)

        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()
        else:
            print("WARNING: Lidar device not found.", file=sys.stderr)

        if self.left_motor:
            self.left_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
        else:
            print("ERROR: Left motor 'wheel_left_joint' not found. Robot cannot move.", file=sys.stderr)

        if self.right_motor:
            self.right_motor.setPosition(float('inf'))
            self.right_motor.setVelocity(0.0)
        else:
            print("ERROR: Right motor 'wheel_right_joint' not found. Robot cannot move.", file=sys.stderr)

        if not self.display:
            print("WARNING: Display device 'display' not found. Trajectory visualization will not work.", file=sys.stderr)
        #if not self.map_display:
        #    print("WARNING: Display device 'map_display' not found. Map visualization will not work. Please add a Display node named 'map_display' to your robot.", file=sys.stderr)
        else:
            pass # self.map_display.setMode(Display.RGB_IMAGE) is not needed here, handled internally

    @staticmethod
    def _calculate_bearing(compass_values):
        x_compass = compass_values[0]
        y_compass = compass_values[1]
        bearing = math.atan2(x_compass, y_compass) # math.atan2 returns angle in radians
        if bearing < 0:
            bearing += 2 * math.pi
        return bearing

    @staticmethod
    def _normalize_angle(angle):
        """Normalizes an angle to be between -PI and PI."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def _get_angle_to_target(current_x, current_y, target_x, target_y):
        """
        Calculates the angle to target in the XY plane relative to the current position.
        This uses math.atan2(dx, dy) to be consistent with _calculate_bearing's
        interpretation of angle relative to the positive Y-axis.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        return math.atan2(dy, dx) # Changed to atan2(dx, dy) for consistency with atan2(compass_x, compass_y)

    @staticmethod
    def _clamp(value, min_val, max_val):
        """Clamps a value between a min and max."""
        return max(min_val, min(value, max_val))

    # --- Trajectory Following ---
    def _follow_trajectory(self):
        if self.trajectory_finished:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return

        current_pos = self.gps.getValues()  # [x, y, z]
        current_x = current_pos[0] # Use X for horizontal
        current_y = current_pos[1] # Use Y for horizontal

        current_waypoint = WAYPOINTS[self.current_waypoint_index]
        self.marker.setSFVec3f([*WAYPOINTS[self.current_waypoint_index], 0])

        distance = math.sqrt(
            (current_x - current_waypoint[0])**2 +
            (current_y - current_waypoint[1])**2 # Use Y for distance calculation
        )

        if distance < WAYPOINT_THRESHOLD:
            print(f"Waypoint {self.current_waypoint_index} reached.")
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(WAYPOINTS):
                print("All waypoints reached. Trajectory finished.")
                self.trajectory_finished = True
                return
            current_waypoint = WAYPOINTS[self.current_waypoint_index]

        compass_values = self.compass.getValues()
        current_heading = self._calculate_bearing(compass_values)

        desired_heading = self._get_angle_to_target(current_x, current_y, current_waypoint[0], current_waypoint[1])

        heading_error = self._normalize_angle(desired_heading - current_heading)
        angular_velocity = self.heading_pid.update(heading_error, self.time_step / 1000.0)

        base_speed = MAX_SPEED * 0.5
        linear_speed = base_speed * min(1.0, distance / WAYPOINT_THRESHOLD)

        left_speed = linear_speed - angular_velocity
        right_speed = linear_speed + angular_velocity

        left_speed = self._clamp(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = self._clamp(right_speed, -MAX_SPEED, MAX_SPEED)

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Display robot path on the Webots 'display' device (not map_display)
        if self.display:
            # Map world coordinates to display pixel coordinates (which might be different than map grid)
            # Assuming the 'display' device is set up with dimensions that cover the robot's world.
            # You might need to adjust these mappings based on your 'display' node setup.
            display_x, display_y = self.occupancy_map._world_to_grid(current_x, current_y)
            display_x = self._clamp(display_x, 0, self.display.getWidth() - 1)
            display_y = self._clamp(display_y, 0, self.display.getHeight() - 1)
            self.display.setColor(0xFF0000) # Red for robot path
            self.display.drawPixel(display_x, display_y)

        current_time = self.robot.getTime() * 1000
        if current_time - self.start_time > TIME_LIMIT_MS:
            print("Time limit (2 minutes) reached. Stopping trajectory following.")
            self.trajectory_finished = True

    def _display_webots_map(self, map_data):
        """
        Displays the given map on the Webots 'display' device with scaling
        to fill the entire display area, regardless of map resolution.
        """
        if not self.display:
            return
        
        # 1. Get the actual dimensions of the Webots Display node
        display_width = self.display.getWidth()
        display_height = self.display.getHeight()

        # 2. Calculate scaling factors
        # These factors determine how many display pixels correspond to one map grid cell
        scale_x = display_width / self.occupancy_map.GRID_WIDTH
        scale_y = display_height / self.occupancy_map.GRID_HEIGHT

        # 3. Clear the display before drawing the new map frame
        self.display.setColor(0x000000) # Black background
        self.display.fillRectangle(0, 0, display_width, display_height)

        # 4. Iterate through grid cells and draw scaled rectangles
        for y_grid in range(self.occupancy_map.GRID_HEIGHT): # Iterate through Y-dimension (rows)
            for x_grid in range(self.occupancy_map.GRID_WIDTH): # Iterate through X-dimension (columns)
                # Calculate pixel color based on map data (same as before)
                if map_data.dtype == np.float32: # Probabilistic map
                    log_odds = map_data[y_grid, x_grid] # Access map using (col, row) -- wrong
                    probability = self.occupancy_map._log_odds_to_probability(log_odds)
                    gray_value = int(probability * 255)
                    color = (gray_value << 16) | (gray_value << 8) | gray_value
                else: # Binary map (C-Space)
                    is_obstacle = map_data[y_grid, x_grid] # Access map using (col, row) -- wrong
                    color = 0x000000 if is_obstacle else 0xFFFFFF # Black for obstacles, white for free
                
                self.display.setColor(color)

                # 5. Calculate the top-left corner of the rectangle on the display
                #    The Y-axis is flipped because Webots Display (0,0) is top-left,
                #    while our map's (0,0) is conceptualized as bottom-left.
                display_rect_x = int(x_grid * scale_x)
                #display_rect_y = int((self.occupancy_map.GRID_HEIGHT - 1 - y_grid) * scale_y)
                display_rect_y = int(y_grid * scale_y)

                # Calculate the width and height of the rectangle
                # IMPORTANT CHANGE: Use math.ceil() to ensure coverage
                rect_width = int(math.ceil(scale_x))
                rect_height = int(math.ceil(scale_y))
                
                # 7. Ensure minimum 1 pixel size if scaling is very small
                #    This prevents issues if scale_x or scale_y results in 0 (e.g., if display is smaller than map)
                if rect_width == 0: rect_width = 1
                if rect_height == 0: rect_height = 1

                # 8. Draw the scaled rectangle instead of a single pixel
                self.display.fillRectangle(display_rect_x, display_rect_y, rect_width, rect_height)

    # --- Main Robot Loop ---
    def run(self):
        """Main loop function for the robot controller."""
        while self.robot.step(self.time_step) != -1:
            
            if not self.trajectory_finished:
                self._follow_trajectory()
                
                self.timestep_counter += 1
                if self.timestep_counter % self.path_interval == 0:
                    robot_position = self.gps.getValues()
                    self.path.append((float(robot_position[0]), float(robot_position[1])))
                
                robot_pos = self.gps.getValues()
                robot_x = robot_pos[0]
                robot_y = robot_pos[1]
                robot_heading = self._calculate_bearing(self.compass.getValues())

                range_image = self.lidar.getRangeImage()
                horizontal_resolution = self.lidar.getHorizontalResolution()
                lidar_points = self.lidar.getNumberOfPoints()
                field_of_view = self.lidar.getFov()
                max_range = self.lidar.getMaxRange()
                
                self.occupancy_map.update_map2(
                    robot_x, robot_y, robot_heading,
                    range_image, horizontal_resolution, 
                    field_of_view, max_range
                )
                self._display_webots_map(self.occupancy_map.get_map_data())
            else:
                if not self.map_processed_and_saved:
                    print("Trajectory finished. Processing and saving maps...")
                    
                    self.occupancy_map.save_map_as_image(
                        self.occupancy_map.get_map_data(),
                        "Final Probabilistic Map",
                        "probabilistic_map"
                    )

                    grown_map = self.occupancy_map.grow_obstacles()
                    self.occupancy_map.save_map_as_image(grown_map, "Grown Probabilistic Map", "grown_map_probabilistic")
                    
                    c_space = self.occupancy_map.compute_c_space(grown_map)
                    self.occupancy_map.save_map_as_image(c_space, "Configuration Space", "c_space_map", cmap='binary')

                    self._display_webots_map(c_space)
                    
                    print("All maps processed and saved. Robot stopped.")
                    self.left_motor.setVelocity(0)
                    self.right_motor.setVelocity(0)
                    self.map_processed_and_saved = True


controller = TaigoliteController(robot_radius=0.2) # Pass the robot radius to the controller
controller.run()