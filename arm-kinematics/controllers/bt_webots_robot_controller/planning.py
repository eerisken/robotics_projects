# planning.py
import py_trees
import numpy as np
import heapq # For A* priority queue
import matplotlib
matplotlib.use('Agg') # Set the backend to 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os # Import os for path joining
from scipy.ndimage import distance_transform_edt # Added for safety cost calculation

# Import the Map class
from map_data import Map

class AStarPlanner:
    """
    Implements the A* pathfinding algorithm on an occupancy grid map.
    Prioritizes safer paths by penalizing proximity to obstacles.
    """
    def __init__(self, map_object: Map): # Now takes a Map object
        self.map = map_object # Store the Map object
        self.grid = self.map.grid # Get the grid array
        self.rows, self.cols = self.grid.shape
        self.map_resolution = self.map.map_resolution
        self.map_origin_x = self.map.map_origin_x
        self.map_origin_y = self.map.map_origin_y

        # Parameters for safety cost
        self.safety_weight = 0.5 # Tune this value: higher means more emphasis on safety
                                 # (can make paths longer, but safer)
        
        # Calculate distance transform for safety cost
        # distance_transform_edt treats 0 as features (obstacles) and 1 as background (free space)
        # Our grid: 2=occupied, 0/1=free. So, invert 2 to 0, and 0/1 to 1.
        binary_c_space_for_dist_transform = (self.grid != Map.OCCUPIED).astype(np.uint8) # 1 for free, 0 for occupied
        self.distance_to_obstacles_map = distance_transform_edt(binary_c_space_for_dist_transform)
        
        # Get the maximum distance to normalize safety cost (avoid division by zero)
        self.max_obstacle_distance = np.max(self.distance_to_obstacles_map)
        if self.max_obstacle_distance == 0:
            self.max_obstacle_distance = 1.0 # Prevent division by zero if map is entirely occupied/empty

    def _is_valid(self, r, c):
        """Checks if a cell is within grid boundaries and not an obstacle."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] != Map.OCCUPIED

    def _heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _check_line_collision(self, p1_global, p2_global):
        """
        Checks if the straight line segment between two global coordinates
        is collision-free in the occupancy grid.
        p1_global, p2_global are [x, y] global coordinates.
        """
        # Use Map object's world_to_pixel and get_cell_state methods
        p1_row, p1_col = self.map._world_to_pixel(p1_global[0], p1_global[1])
        p2_row, p2_col = self.map._world_to_pixel(p2_global[0], p2_global[1])

        # Use a simple line drawing algorithm (Bresenham-like) to check pixels
        num_steps = max(abs(p1_col - p2_col), abs(p1_row - p2_row)) + 1
        if num_steps == 1: # If points are the same or adjacent, just check the end point
            return self._is_valid(p2_row, p2_col)

        for i in range(num_steps):
            t = i / (num_steps - 1)
            current_col = int(p1_col + t * (p2_col - p1_col))
            current_row = int(p1_row + t * (p2_row - p1_row))

            if not self._is_valid(current_row, current_col):
                return False # Collision detected

        return True # No collision along the line segment

    def find_path(self, start_pixel, goal_pixel):
        """
        Finds the shortest path from start_pixel to goal_pixel using A*.
        start_pixel and goal_pixel are (row, col) tuples.
        Returns a list of (x, y) global coordinates representing the path.
        """
        if not self._is_valid(start_pixel[0], start_pixel[1]) or \
           not self._is_valid(goal_pixel[0], goal_pixel[1]):
            print(f"AStarPlanner: Start or goal pixel is invalid or occupied.")
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_pixel)) # (f_score, (row, col))

        came_from = {} # (row, col) -> (row, col)

        g_score = { (r, c): float('inf') for r in range(self.rows) for c in range(self.cols) }
        g_score[start_pixel] = 0

        f_score = { (r, c): float('inf') for r in range(self.rows) for c in range(self.cols) }
        f_score[start_pixel] = self._heuristic(start_pixel, goal_pixel)

        # 8-directional movement (including diagonals)
        # (dr, dc) for (row_change, col_change)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1), # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1) # Diagonal
        ]
        # Cost for cardinal moves
        cost_cardinal = 1.0
        # Cost for diagonal moves is sqrt(2) times cardinal cost
        cost_diagonal = np.sqrt(2)

        while open_set:
            current_f, current_pixel = heapq.heappop(open_set)

            if current_pixel == goal_pixel:
                # Reconstruct path
                path_pixels = []
                while current_pixel in came_from:
                    path_pixels.append(current_pixel)
                    current_pixel = came_from[current_pixel]
                path_pixels.append(start_pixel)
                path_pixels.reverse()
                
                # Convert pixel path to global coordinates and round to 2 decimal places
                path_global_coords = []
                for r, c in path_pixels:
                    global_x, global_y = self.map._pixel_to_world(r, c)
                    path_global_coords.append([round(global_x, 2), round(global_y, 2)])
                
                return path_global_coords

            for i, (dr, dc) in enumerate(directions):
                neighbor_pixel = (current_pixel[0] + dr, current_pixel[1] + dc)

                if not self._is_valid(neighbor_pixel[0], neighbor_pixel[1]):
                    continue

                # Determine base movement cost
                move_cost = cost_cardinal if i < 4 else cost_diagonal
                
                # Calculate safety cost based on distance to nearest obstacle
                distance_val = self.distance_to_obstacles_map[neighbor_pixel[0], neighbor_pixel[1]]
                
                # Safety cost: penalize cells closer to obstacles.
                # Max distance (safest) should add 0 safety cost.
                # Min distance (closest to obstacle, but still valid) should add max safety cost.
                safety_cost = self.safety_weight * (self.max_obstacle_distance - distance_val)
                safety_cost = max(0.0, safety_cost) # Ensure safety cost is non-negative

                tentative_g_score = g_score[current_pixel] + move_cost + safety_cost

                if tentative_g_score < g_score[neighbor_pixel]:
                    came_from[neighbor_pixel] = current_pixel
                    g_score[neighbor_pixel] = tentative_g_score
                    f_score[neighbor_pixel] = tentative_g_score + self._heuristic(neighbor_pixel, goal_pixel)
                    if (f_score[neighbor_pixel], neighbor_pixel) not in open_set: # Check if already in heap (less efficient, but safer)
                        heapq.heappush(open_set, (f_score[neighbor_pixel], neighbor_pixel))
        
        print(f"AStarPlanner: No path found from {start_pixel} to {goal_pixel}.")
        return [] # No path found

    def smooth_path(self, path, iterations=100):
        """
        Applies a simple iterative path smoothing algorithm.
        Tries to connect non-adjacent waypoints directly if the line segment is collision-free.
        :param path: List of global [x, y] coordinates from A*.
        :param iterations: Number of smoothing iterations.
        :return: Smoothed path (list of global [x, y] coordinates).
        """
        if not path or len(path) < 3: # Need at least 3 points to smooth
            return path

        smoothed_path = list(path) # Start with a copy of the original path

        for _ in range(iterations):
            # Pick two random distinct indices
            # Ensure indices are within the current length of the smoothed_path
            if len(smoothed_path) < 2: # If path becomes too short, stop smoothing
                break
            idx1, idx2 = np.random.choice(len(smoothed_path), 2, replace=False)
            if idx1 > idx2: # Ensure idx1 is always smaller
                idx1, idx2 = idx2, idx1
            
            if idx2 - idx1 <= 1: # Cannot smooth if points are adjacent
                continue

            p1 = smoothed_path[idx1]
            p2 = smoothed_path[idx2]

            # Check if the straight line segment between p1 and p2 is collision-free
            if self._check_line_collision(p1, p2):
                # If collision-free, remove intermediate points and replace with p1, p2
                # This creates a new path segment
                new_segment = [p1]
                if p1 != p2: # Only add p2 if it's different from p1
                    new_segment.append(p2)
                
                # Construct the new path
                smoothed_path = smoothed_path[:idx1] + new_segment + smoothed_path[idx2+1:]
        
        return smoothed_path

class ComputePathToGoal(py_trees.behaviour.Behaviour):
    """
    Abstract base class for computing a path to a specific goal.
    Subclasses will define the specific goal coordinates.
    """
    def __init__(self, name, blackboard, goal_coords_key):
        super(ComputePathToGoal, self).__init__(name)
        self.blackboard = blackboard
        self.goal_coords_key = goal_coords_key # Key in blackboard for goal coordinates (e.g., 'goal_lower_left')

    def initialise(self):
        """
        Loads the C-space, defines start and goal, computes path using A*,
        smooths the path, and stores the path on the blackboard.
        """
        # Ensure C-space map object is available
        if self.blackboard.c_space_map_object is None or self.blackboard.c_space_map_object.grid is None:
            self.logger.error(f"{self.name}: C-space map object not available or empty on blackboard. Cannot plan path.")
            return py_trees.common.Status.FAILURE

        c_space_map = self.blackboard.c_space_map_object # Get the C-space Map object
        
        robot_x = self.blackboard.robot_pose['x']
        robot_y = self.blackboard.robot_pose['y'] # This is the Z from GPS in Webots
        
        self.blackboard.current_goal = self.goal_coords_key
        goal_x, goal_y = getattr(self.blackboard, self.goal_coords_key)

        # Convert global coordinates to map pixel coordinates using the Map object's method
        start_pixel_row, start_pixel_col = c_space_map._world_to_pixel(robot_x, robot_y)
        goal_pixel_row, goal_pixel_col = c_space_map._world_to_pixel(goal_x, goal_y)

        start_pixel = (start_pixel_row, start_pixel_col)
        goal_pixel = (goal_pixel_row, goal_pixel_col)

        # Ensure start and goal are within map bounds
        map_rows, map_cols = c_space_map.rows, c_space_map.cols
        if not (0 <= start_pixel_row < map_rows and 0 <= start_pixel_col < map_cols and \
                0 <= goal_pixel_row < map_rows and 0 <= goal_pixel_col < map_cols):
            self.logger.error(f"{self.name}: Start or goal pixel is out of map bounds. Start: {start_pixel}, Goal: {goal_pixel}. Map dims: {map_rows}x{map_cols}")
            # Plot the C-space with start/goal if out of bounds
            self._plot_failed_path(c_space_map, start_pixel, goal_pixel, "Start or Goal Out of Bounds")
            return py_trees.common.Status.FAILURE
        
        # Check if start or goal are in occupied cells in C-space
        if c_space_map.grid[start_pixel[0], start_pixel[1]] == Map.OCCUPIED or \
           c_space_map.grid[goal_pixel[0], goal_pixel[1]] == Map.OCCUPIED:
            self.logger.error(f"{self.name}: Start or goal pixel is in an occupied cell in C-space. Start: {start_pixel}, Goal: {goal_pixel}")
            # Plot the C-space with start/goal if in occupied cell
            self._plot_failed_path(c_space_map, start_pixel, goal_pixel, "Start or Goal in Occupied Cell")
            return py_trees.common.Status.FAILURE


        self.logger.info(f"{self.name}: Planning path from robot pose ({robot_x:.2f}, {robot_y:.2f}) to goal ({goal_x:.2f}, {goal_y:.2f})")
        self.logger.info(f"{self.name}: Pixels: Start {start_pixel}, Goal {goal_pixel}")

        planner = AStarPlanner(c_space_map) # Pass the C-space Map object
        raw_path = planner.find_path(start_pixel, goal_pixel)

        if raw_path:
            self.logger.info(f"{self.name}: Raw path found with {len(raw_path)} waypoints. Smoothing path...")
            # Pass a higher number of iterations for smoothing to reduce waypoints
            smoothed_path = raw_path #planner.smooth_path(raw_path, iterations=100) # Increased iterations for better smoothing
            
            self.blackboard.waypoints_to_follow = smoothed_path
            self.logger.info(f"{self.name}: Smoothed path has {len(smoothed_path)} waypoints.")
            
            # Plot the successful path
            self._plot_successful_path(c_space_map, start_pixel, goal_pixel, raw_path, smoothed_path, "Successful Path")
            
            return py_trees.common.Status.SUCCESS
        else:
            self.blackboard.waypoints_to_follow = []
            self.logger.warning(f"{self.name}: No path found to goal.")
            # Plot the C-space with start/goal if no path found
            self._plot_failed_path(c_space_map, start_pixel, goal_pixel, "No Path Found")
            return py_trees.common.Status.FAILURE

    def update(self):
        """This behaviour is instantaneous. It returns SUCCESS if a path has been computed
        and stored on the blackboard, otherwise FAILURE."""
        if self.blackboard.waypoints_to_follow:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(f"{self.name}: Terminated with status {new_status}.")

    def _plot_failed_path(self, c_space_map: Map, start_pixel: tuple, goal_pixel: tuple, title_suffix: str = ""):
        """
        Helper function to plot the C-space with start and goal pixels
        when path planning fails.
        """
        try:
            plt.figure(figsize=(8, 8))
            cmap = ListedColormap(['lightgray', 'lightgray', 'red']) # 0: Free, 1: Free, 2: Occupied
            
            # Plot the C-space using the Map object's grid and properties
            plt.imshow(c_space_map.grid, cmap=cmap, vmin=0, vmax=2, origin='lower',
                       extent=[c_space_map.map_origin_x, c_space_map.map_origin_x + c_space_map.map_dimension_x,
                               c_space_map.map_origin_y, c_space_map.map_origin_y + c_space_map.map_dimension_y])

            # Plot start pixel (green dot)
            start_x_global, start_y_global = c_space_map._pixel_to_world(start_pixel[0], start_pixel[1])
            plt.plot(start_x_global, start_y_global, 'go', markersize=10, label='Start') 

            # Plot goal pixel (blue star)
            goal_x_global, goal_y_global = c_space_map._pixel_to_world(goal_pixel[0], goal_pixel[1])
            plt.plot(goal_x_global, goal_y_global, 'b*', markersize=12, label='Goal') 

            plt.title(f"C-Space with Start/Goal ({title_suffix})")
            plt.xlabel(f"X (meters)")
            plt.ylabel(f"Y (meters)")
            plt.colorbar(ticks=[0, 1, 2], label="Map State (0:Free, 1:Free, 2:Occupied)")
            plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
            plt.legend()
            
            # Save the figure to a file in the map-outputs directory
            output_dir = self.blackboard.output_dir
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.join(output_dir, f"path_planning_failure_{title_suffix.replace(' ', '_').lower()}.png")
            plt.savefig(file_name)
            self.logger.info(f"{self.name}: Path planning failure plot saved to {file_name}")
            plt.close() # Close the plot to free memory

        except Exception as e:
            self.logger.error(f"{self.name}: Error plotting failed path: {e}")

    def _plot_successful_path(self, c_space_map: Map, start_pixel: tuple, goal_pixel: tuple, raw_path: list, smoothed_path: list, title_suffix: str = ""):
        """
        Helper function to plot the C-space with start, goal, raw path, and smoothed path
        when path planning is successful.
        """
        try:
            plt.figure(figsize=(8, 8))
            cmap = ListedColormap(['lightgray', 'lightgray', 'red']) # 0: Free, 1: Free, 2: Occupied
            
            # Plot the C-space using the Map object's grid and properties
            plt.imshow(c_space_map.grid, cmap=cmap, vmin=0, vmax=2, origin='lower',
                       extent=[c_space_map.map_origin_x, c_space_map.map_origin_x + c_space_map.map_dimension_x,
                               c_space_map.map_origin_y, c_space_map.map_origin_y + c_space_map.map_dimension_y])

            # Plot start pixel (green dot)
            start_x_global, start_y_global = c_space_map._pixel_to_world(start_pixel[0], start_pixel[1])
            plt.plot(start_x_global, start_y_global, 'go', markersize=10, label='Start') 
            
            # Plot goal pixel (blue star)
            goal_x_global, goal_y_global = c_space_map._pixel_to_world(goal_pixel[0], goal_pixel[1])
            plt.plot(goal_x_global, goal_y_global, 'b*', markersize=12, label='Goal') 

            # Plot raw path (thin orange line)
            if raw_path:
                raw_path_x = [p[0] for p in raw_path]
                raw_path_y = [p[1] for p in raw_path]
                plt.plot(raw_path_x, raw_path_y, 'orange', linestyle='--', linewidth=1, label='Raw Path')

            # Plot smoothed path (thick cyan line)
            if smoothed_path:
                smoothed_path_x = [p[0] for p in smoothed_path]
                smoothed_path_y = [p[1] for p in smoothed_path]
                plt.plot(smoothed_path_x, smoothed_path_y, 'cyan', linewidth=3, label='Smoothed Path') 

            plt.title(f"C-Space with Path ({title_suffix})")
            plt.xlabel(f"X (meters)")
            plt.ylabel(f"Y (meters)")
            plt.colorbar(ticks=[0, 1, 2], label="Map State (0:Free, 1:Free, 2:Occupied)")
            plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
            plt.legend()
            
            # Save the figure to a file in the map-outputs directory
            output_dir = self.blackboard.output_dir
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.join(output_dir, f"path_planning_success_{title_suffix.replace(' ', '_').lower()}.png")
            plt.savefig(file_name)
            self.logger.info(f"{self.name}: Path planning success plot saved to {file_name}")
            plt.close() # Close the plot to free memory

        except Exception as e:
            self.logger.error(f"{self.name}: Error plotting successful path: {e}")

class ComputePathToLowerLeftCorner(ComputePathToGoal):
    """Computes path to the lower-left corner goal."""
    def __init__(self, blackboard):
        super(ComputePathToLowerLeftCorner, self).__init__("Compute Path to Lower Left Corner", blackboard, 'goal_lower_left')

class ComputePathToSink(ComputePathToGoal):
    """Computes path to the sink goal."""
    def __init__(self, blackboard):
        super(ComputePathToSink, self).__init__("Compute Path to Sink", blackboard, 'goal_sink')
