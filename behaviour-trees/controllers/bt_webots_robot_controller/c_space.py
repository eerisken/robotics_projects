# c_space_generator.py
import numpy as np
import os
from scipy.ndimage import binary_dilation, median_filter, uniform_filter, distance_transform_edt
from map_data import Map # Import the Map class

class CSpaceGenerator:
    """
    A class to generate a Configuration Space (C-Space) map from an
    occupancy grid map. It applies noise reduction and obstacle dilation.
    """
    def __init__(self, map_object: Map, robot_radius_pixels: int = 6, filter_size: int = 3):
        """
        Initializes the CSpaceGenerator.

        :param map_object: An instance of the Map class containing the occupancy grid.
        :param robot_radius_pixels: The approximate radius of the robot in map pixels,
                                    used for obstacle dilation.
        :param filter_size: The size of the filter kernel (e.g., 3 for 3x3)
                                   for noise reduction.
        """
        if not isinstance(map_object, Map):
            raise TypeError("map_object must be an instance of the Map class.")
            
        self.original_map = map_object
        self.c_space_grid = None # This will store the generated C-space NumPy array
        self.robot_radius_pixels = robot_radius_pixels
        self.filter_size = filter_size

        # Create a separate Map object for the C-space to leverage its plotting/saving methods
        # It will initially have the same dimensions and origin as the original map
        self.c_space_map_object = Map(
            map_origin_x=self.original_map.map_origin_x,
            map_origin_y=self.original_map.map_origin_y,
            map_resolution=self.original_map.map_resolution,
            map_dimension_x=self.original_map.map_dimension_x,
            map_dimension_y=self.original_map.map_dimension_y,
            output_directory=self.original_map.output_dir # Use the same output directory
        )
        # Suppress initial print from C-space map object to avoid clutter
        self.c_space_map_object.grid = np.zeros_like(self.original_map.grid) # Ensure it's empty initially

        print(f"CSpaceGenerator initialized with robot_radius_pixels={self.robot_radius_pixels}, "
              f"filter_size={self.filter_size}.")
        
    def generate_c_space(self):
        """
        Generates the configuration space map from the original occupancy grid.
        Applies filtering for noise reduction and then dilates obstacles.
        """
        if self.original_map.grid is None:
            print("Error: Original map grid is empty. Cannot generate C-space.")
            return

        # Step 1: Apply filter to the raw occupancy grid to reduce noise
        # Convert to binary for filter: 1 for occupied (2), 0 for free (0,1)
        binary_map_for_filter = (self.original_map.grid == Map.OCCUPIED).astype(np.uint8)
        
        # Apply filter
        filtered_map_binary = uniform_filter(binary_map_for_filter, size=self.filter_size)

        # Initialize C-space grid with original map values, then update based on filtered obstacles
        c_space_temp = np.copy(self.original_map.grid)
        # Set all originally occupied cells to free (1) temporarily, then mark filtered obstacles as occupied (2)
        c_space_temp[c_space_temp == Map.OCCUPIED] = Map.FREE
        c_space_temp[filtered_map_binary == 1] = Map.OCCUPIED # Mark as occupied where filter says it's an obstacle.

        # Step 2: Dilate obstacles in the filtered map to create the C-space
        dilated_c_space = np.copy(c_space_temp) # Start with the noise-reduced map for dilation

        rows, cols = self.original_map.rows, self.original_map.cols

        # Iterate through each cell and dilate if it's an obstacle
        for r in range(rows):
            for c in range(cols):
                if c_space_temp[r, c] == Map.OCCUPIED: # If occupied (after filtering)
                    # Dilate around this occupied cell
                    for dr in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                        for dc in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                            nr, nc = r + dr, c + dc
                            # Check bounds and ensure we don't overwrite existing obstacles (Map.OCCUPIED)
                            if 0 <= nr < rows and 0 <= nc < cols and dilated_c_space[nr, nc] != Map.OCCUPIED:
                                dilated_c_space[nr, nc] = Map.OCCUPIED # Mark as occupied in C-space

        self.c_space_grid = dilated_c_space
        self.c_space_map_object.grid = self.c_space_grid # Update the C-space Map object's grid
        print("C-space generated successfully.")

    def save_and_plot_maps(self, map_file_name: str = "kitchen_map.npy", 
                           c_space_file_name: str = "kitchen_c_space.npy",
                           map_plot_name: str = "final_occupancy_map.png",
                           c_space_plot_name: str = "final_c_space_map.png"):
        """
        Saves the original map and the generated C-space grid data to .npy files
        and plots them as PNG images in the output directory.

        :param map_file_name: Filename for the original map .npy data.
        :param c_space_file_name: Filename for the C-space .npy data.
        :param map_plot_name: Filename for the original map plot .png.
        :param c_space_plot_name: Filename for the C-space plot .png.
        """
        # Save and plot original map
        if self.original_map.grid is not None:
            self.original_map.save_map_data(map_file_name)
            self.original_map.plot_map(map_plot_name, "Generated Occupancy Grid Map")
        else:
            print("Original map grid is empty, skipping save/plot for original map.")

        # Save and plot C-space map
        if self.c_space_grid is not None:
            self.c_space_map_object.save_map_data(c_space_file_name)
            self.c_space_map_object.plot_map(c_space_plot_name, "Generated Configuration Space Map (C-Space)")
        else:
            print("C-space grid is empty, skipping save/plot for C-space.")

