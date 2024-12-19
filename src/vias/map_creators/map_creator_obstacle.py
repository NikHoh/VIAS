from math import ceil

import numpy as np
from scipy.ndimage import binary_dilation

from src.vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator
from vias.console_manager import console


class MapCreatorObstacle(MapCreator):
    def create_map(self, map_name) -> GridMap:
        buildings_map: GridMap = self.inputs["buildings_map"]
        horizontal_safety_distance = self.parameters["horizontal_safety_distance"]
        vertical_safety_distance = self.parameters["vertical_safety_distance"]
        obstacle_map = self._get_map_blueprint(map_name, 3)
        console.log("Map {} is being rendered. This may take a while ...".format(obstacle_map.name))
        # iterate over the buildings grid_map having the height of the buildings
        for index, height in np.ndenumerate(buildings_map.grid_array):
            # discretize height to layer index
            assert height >= 0.0, "Height is negative, this is not allowed"
            lay_idx = np.ceil(height / self.scenario_info.z_res).astype(int)
            obstacle_map.grid_tensor[index[0], index[1], 0:lay_idx] = 1.0 # set all elements until height index to 1.0

        # dilate the obstacle grid_map with respect to the safety distances

        structure_x_dim = ceil(horizontal_safety_distance / self.scenario_info.x_res)
        structure_y_dim = ceil(horizontal_safety_distance / self.scenario_info.y_res)
        structure_z_dim = ceil(vertical_safety_distance / self.scenario_info.z_res)

        obstacle_map.grid_tensor = binary_dilation(obstacle_map.grid_tensor, structure=np.ones((structure_y_dim, structure_x_dim, structure_z_dim)), iterations=1)

        return obstacle_map