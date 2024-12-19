import numpy as np

from src.vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator


class MapCreatorClearanceHeight(MapCreator):
    def create_map(self, map_name) -> GridMap:
        buildings_map = self.inputs["buildings_map"]
        vertical_safety_distance = self.parameters["vertical_safety_distance"]
        clearance_height_map = self._get_map_blueprint(map_name, 2)
        clearance_height_map.set_from_array(buildings_map.grid_array)




        safety_distance_grid = np.ones(clearance_height_map.grid_array.shape) * vertical_safety_distance

        clearance_height_map.grid_array[clearance_height_map.grid_array != 0] += safety_distance_grid[clearance_height_map.grid_array != 0]

        return clearance_height_map