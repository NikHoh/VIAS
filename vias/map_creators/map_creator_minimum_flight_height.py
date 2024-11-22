import numpy as np

from vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator


class MapCreatorMinimumFlightHeight(MapCreator):
    def create_map(self, map_name) -> GridMap:
        minimum_flight_height_map = self._get_map_blueprint(map_name, 3)

        # create a min flight array indicating all cells below the min flight height with one
        min_flight_array = np.zeros(minimum_flight_height_map.grid_tensor.shape)
        lay_idx = np.ceil(self.scenario_info.min_flight_height / self.scenario_info.z_res).astype(int)
        min_flight_array[:, :, 0:lay_idx] = 1.0  # set all elements below lay idx to 1.0

        minimum_flight_height_map.set_from_array(min_flight_array)

        return minimum_flight_height_map