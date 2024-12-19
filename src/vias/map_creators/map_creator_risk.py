import numpy as np

from vias.console_manager import console
from vias.grid_map import GridMap
from vias.map_creators.map_creator import (
    MapCreator,
    get_distances_to_nonzero_cells,
    thin_nonzero_cells,
)


class MapCreatorRisk(MapCreator):
    def create_map(self, map_name) -> GridMap:
        buildings_map: GridMap = self.inputs["buildings_map"]
        semantic_map = self.inputs["semantic_map"]
        min_flight_height = self.scenario_info.min_flight_height
        clearance_height_map = self.inputs["clearance_height_map"]
        risk_map = self._get_map_blueprint(map_name, 3)
        console.log(f"Map {risk_map.name} is being rendered. This may take a while ...")

        cells_with_low_risk = np.zeros(risk_map.grid_tensor.shape)

        # it is less risky to fly in the middle over buildings
        thinned_buildings = thin_nonzero_cells(buildings_map.grid_array)
        for index, value in np.ndenumerate(thinned_buildings):
            if value == 1.0:
                corresponding_clearance_height = clearance_height_map.grid_array[
                    index[0], index[1]
                ]
                height_to_fly = max(corresponding_clearance_height, min_flight_height)
                lay_idx = np.floor(height_to_fly / self.scenario_info.z_res).astype(int)
                if (
                    lay_idx <= cells_with_low_risk.shape[2] - 1
                ):  # otherwise the building is too high
                    cells_with_low_risk[index[0], index[1], lay_idx] = 1.0

        # it is less risky to fly in the middle over water areas
        water_array = np.zeros(semantic_map.grid_array.shape)
        water_array[semantic_map.grid_array == 249.6] = 1.0  # set areas of water to 1
        thinned_water = thin_nonzero_cells(water_array)
        for index, value in np.ndenumerate(thinned_water):
            if value == 1.0:
                lay_idx = np.floor(min_flight_height / self.scenario_info.z_res).astype(
                    int
                )
                if lay_idx <= cells_with_low_risk.shape[2] - 1:
                    cells_with_low_risk[index[0], index[1], lay_idx] = 1.0

        # now all cells that are of low risk are one, all others are zero,
        # create Euclidean Distance transform
        risk_map.set_from_array(get_distances_to_nonzero_cells(cells_with_low_risk))

        # allow only positive entries
        min_value = np.min(risk_map.grid_tensor)
        if min_value <= 0:
            risk_map.grid_tensor = risk_map.grid_tensor - min_value + 1e-10

        return risk_map
