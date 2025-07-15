# map_class.py
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Set the backend to 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Map:
    """
    A class to represent and manage an occupancy grid map.

    The map uses the following conventions:
    - Grid cell values: 0 (unknown/free), 1 (free), 2 (occupied).
      For plotting, 0 and 1 are treated as free (lightgray).
    - World coordinates (x, y) map to (column, row) in the NumPy array.
    - The (0,0) index of the NumPy array corresponds to (map_origin_x, map_origin_y)
      in world coordinates (bottom-left corner of the map).
    """
    FREE = 1  # Represents free space (or unknown, which is treated as free)
    OCCUPIED = 2 # Represents occupied space

    def __init__(self, map_origin_x: float, map_origin_y: float, 
                 map_resolution: float, map_dimension_x: float, map_dimension_y: float,
                 output_directory: str = "map_outputs"): # Added output_directory parameter
        """
        Initializes the Map.

        :param map_origin_x: X-coordinate of the map's bottom-left corner in global meters.
        :param map_origin_y: Y-coordinate of the map's bottom-left corner in global meters.
        :param map_resolution: Meters per grid cell (e.g., 0.05 for 5 cm per pixel).
        :param map_dimension_x: Total X dimension of the map in meters.
        :param map_dimension_y: Total Y dimension of the map in meters.
        :param output_directory: The subdirectory name to save map data and plots.
        """
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.map_resolution = map_resolution
        self.map_dimension_x = map_dimension_x
        self.map_dimension_y = map_dimension_y

        # Calculate map dimensions in pixels (rows, columns)
        # NumPy arrays are typically (rows, columns) which corresponds to (Y, X)
        self.rows = int(map_dimension_y / map_resolution)
        self.cols = int(map_dimension_x / map_resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8) # Initialize with 0 (unknown/free)

        self.output_dir = output_directory # Store the output directory
        os.makedirs(self.output_dir, exist_ok=True) # Ensure the directory exists

        print(f"Map initialized: {self.rows}x{self.cols} pixels, "
              f"resolution={self.map_resolution} m/px, "
              f"origin=({self.map_origin_x}, {self.map_origin_y}) m, "
              f"output_dir='{self.output_dir}'")

    def _world_to_pixel(self, world_x: float, world_y: float) -> tuple[int, int]:
        """
        Converts world coordinates (meters) to map pixel coordinates (row, column).
        Returns (row, col) tuple.
        """
        col = int((world_x - self.map_origin_x) / self.map_resolution)
        row = int((world_y - self.map_origin_y) / self.map_resolution)
        return row, col

    def _pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """
        Converts map pixel coordinates (row, column) to world coordinates (meters).
        Returns (world_x, world_y) tuple.
        """
        world_x = col * self.map_resolution + self.map_origin_x
        world_y = row * self.map_resolution + self.map_origin_y
        return world_x, world_y

    def set_cell_state(self, world_x: float, world_y: float, state: int):
        """
        Sets the state of a map cell at given world coordinates.

        :param world_x: X-coordinate in meters.
        :param world_y: Y-coordinate in meters.
        :param state: The new state for the cell (Map.FREE or Map.OCCUPIED).
        """
        row, col = self._world_to_pixel(world_x, world_y)

        if 0 <= row < self.rows and 0 <= col < self.cols:
            if state in [self.FREE, self.OCCUPIED]:
                self.grid[row, col] = state
            else:
                print(f"Warning: Invalid state '{state}'. Use Map.FREE or Map.OCCUPIED.")
        else:
            print(f"Warning: World coordinates ({world_x:.2f}, {world_y:.2f}) are outside map bounds.")

    def get_cell_state(self, world_x: float, world_y: float) -> int | None:
        """
        Gets the state of a map cell at given world coordinates.

        :param world_x: X-coordinate in meters.
        :param world_y: Y-coordinate in meters.
        :return: The state of the cell (0, 1, or 2), or None if out of bounds.
        """
        row, col = self._world_to_pixel(world_x, world_y)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col]
        return None

    def save_map_data(self, file_name: str):
        """
        Saves the map's NumPy array data to a .npy file within the output directory.

        :param file_name: The name of the .npy file (e.g., "my_map.npy").
        """
        file_path = os.path.join(self.output_dir, file_name)
        np.save(file_path, self.grid)
        print(f"Map data saved to {file_path}")

    def load_map_data(self, file_name: str) -> bool:
        """
        Loads map data from a .npy file within the output directory into the map's grid.
        Updates map dimensions if loaded map has different dimensions.

        :param file_name: The name of the .npy file (e.g., "my_map.npy").
        :return: True if successful, False otherwise.
        """
        file_path = os.path.join(self.output_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Error: Map file not found at {file_path}")
            return False
        try:
            loaded_grid = np.load(file_path)
            if loaded_grid.dtype != np.int8:
                print(f"Warning: Loaded map data has dtype {loaded_grid.dtype}, converting to int8.")
                loaded_grid = loaded_grid.astype(np.int8)

            self.grid = loaded_grid
            self.rows, self.cols = self.grid.shape
            # Recalculate dimensions in meters based on loaded grid and current resolution
            self.map_dimension_x = self.cols * self.map_resolution
            self.map_dimension_y = self.rows * self.map_resolution
            print(f"Map data loaded from {file_path}. New dimensions: {self.rows}x{self.cols} pixels.")
            return True
        except Exception as e:
            print(f"Error loading map data from {file_path}: {e}")
            return False

    def plot_map(self, file_name: str, title: str = "Occupancy Grid Map"):
        """
        Plots the map and saves it as a PNG image within the output directory.

        :param file_name: The name of the .png file (e.g., "my_map_plot.png").
        :param title: The title for the plot.
        """
        file_path = os.path.join(self.output_dir, file_name)
        
        plt.figure(figsize=(8, 8))
        
        # Create a custom colormap: 0 and 1 are lightgray (free), 2 is red (occupied)
        cmap = ListedColormap(['lightgray', 'lightgray', 'red']) 
        
        # Define extent for correct world coordinate display
        # (xmin, xmax, ymin, ymax)
        extent = [self.map_origin_x, self.map_origin_x + self.map_dimension_x,
                  self.map_origin_y, self.map_origin_y + self.map_dimension_y]

        # Use origin='lower' as (0,0) pixel corresponds to (map_origin_x, map_origin_y)
        plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=extent)

        plt.title(title)
        plt.xlabel(f"X (meters)")
        plt.ylabel(f"Y (meters)")
        plt.colorbar(ticks=[0, 1, 2], label="Map State (0:Free, 1:Free, 2:Occupied)")
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        plt.savefig(file_path)
        print(f"Map plot saved to {file_path}")
        plt.close() # Close the plot to free memory

