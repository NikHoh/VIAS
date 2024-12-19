import numpy as np

from vias.console_manager import console
from vias.grid_map import GridMap
from vias.map_creators.map_creator import (
    MapCreator,
    get_distances_to_nonzero_cells,
    thin_nonzero_cells,
)


class MapCreatorNoise(MapCreator):
    def create_map(self, map_name) -> GridMap:
        streets_map = self.inputs["streets_map"]
        semantic_map = self.inputs["semantic_map"]
        clearance_height_map = self.inputs["clearance_height_map"]
        max_flight_height = self.scenario_info.max_flight_height

        noise_map = self._get_map_blueprint(map_name, 3)
        console.log(
            f"Map {noise_map.name} is being rendered. This may take a while ..."
        )

        cells_with_low_noise = np.zeros(noise_map.shape)

        # it is less noisy to fly over streets in the maximum_flight_height
        thinned_paths = thin_nonzero_cells(streets_map.grid_array)
        for index, value in np.ndenumerate(thinned_paths):
            if value == 1.0:
                corresponding_clearance_height = clearance_height_map.grid_array[
                    index[0], index[1]
                ]
                if corresponding_clearance_height > max_flight_height:
                    continue
                lay_idx = np.floor(max_flight_height / self.scenario_info.z_res).astype(
                    int
                )
                if (
                    lay_idx <= cells_with_low_noise.shape[2] - 1
                ):  # otherwise the cell is out of bound
                    cells_with_low_noise[index[0], index[1], lay_idx] = 1.0

        # it is less noisy to fly in the middle over water areas
        water_array = np.zeros(semantic_map.grid_array.shape)
        water_array[semantic_map.grid_array == 249.6] = 1.0  # set areas of water to 1
        thinned_water = thin_nonzero_cells(water_array)
        for index, value in np.ndenumerate(thinned_water):
            if value == 1.0:
                corresponding_clearance_height = clearance_height_map.grid_array[
                    index[0], index[1]
                ]
                if corresponding_clearance_height > max_flight_height:
                    continue  # there might be a pool (water area) on a high building :D
                lay_idx = np.floor(max_flight_height / self.scenario_info.z_res).astype(
                    int
                )
                if (
                    lay_idx <= cells_with_low_noise.shape[2] - 1
                ):  # otherwise the cell is out of bound
                    cells_with_low_noise[index[0], index[1], lay_idx] = 1.0

        # now all cells that are of low noise are one, all others are zero,
        # create Euclidean Distance transform
        noise_map.set_from_array(get_distances_to_nonzero_cells(cells_with_low_noise))

        # add park areas on all heights to the nfa grid_map
        cells_to_avoid = np.zeros(noise_map.shape)
        cells_to_avoid[semantic_map.grid_array == 211.2] = 1.0  # park areas

        # respect park areas by superimposing Euclidean Distance transform that
        # leads away from NFA areas
        inverted_cells_to_avoid = np.logical_not(cells_to_avoid)
        nfa_distance_transform = get_distances_to_nonzero_cells(inverted_cells_to_avoid)
        noise_map.add_from_array(nfa_distance_transform)

        # allow only positive entries
        min_value = np.min(noise_map.grid_tensor)
        if min_value <= 0:
            noise_map.grid_tensor = noise_map.grid_tensor - min_value + 1e-10

        return noise_map
