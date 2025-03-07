from dataclasses import astuple

import numpy as np

from vias.console_manager import console
from vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator
from vias.utils.helpers import OutOfOperationSpace, TmercCoord
from vias.utils.math_functions import inverse_square_law_with_offset


class MapCreatorRadioDisturbance(MapCreator):
    def create_map(self, map_name) -> GridMap:
        radio_towers = self.inputs["radio_towers"]
        buildings_map: GridMap = self.inputs["buildings_map"]
        best_signal_strength = self.parameters["best_signal_strength"]
        default_radio_tower_height = self.parameters["default_radio_tower_height"]
        radio_disturbance_map = self._get_map_blueprint(map_name, 3)

        console.log(
            f"Map {radio_disturbance_map.name} is being rendered. This may take a "
            f"while ..."
        )

        xv, yv, zv = astuple(radio_disturbance_map.local_coord_meshgrid)

        map_initialized = False
        for loc in radio_towers:
            tmerc_coord = TmercCoord(loc[0], loc[1])
            local_coord = radio_disturbance_map.local_coord_from_tmerc_coord(
                tmerc_coord, 0.0
            )  # we want to access 2D array, so we set z=0.0
            try:
                array_coord = buildings_map.array_coord_from_local_coord(local_coord)
                radio_tower_height = buildings_map.get_values_from_array_coords(
                    [array_coord]
                ).item()
                radio_tower_height = max(radio_tower_height, default_radio_tower_height)
            except OutOfOperationSpace:
                continue
            single_tower_array = inverse_square_law_with_offset(
                xv,
                yv,
                zv,
                local_coord.x,
                local_coord.y,
                radio_tower_height,
                best_signal_strength,
            )
            if map_initialized:
                radio_disturbance_map.add_from_array(
                    single_tower_array, operation_type="min"
                )
            else:
                radio_disturbance_map.set_from_array(single_tower_array)
                map_initialized = True

        # allow only positive entries
        min_value = np.min(radio_disturbance_map.grid_tensor)
        if min_value <= 0:
            radio_disturbance_map.grid_tensor = (
                radio_disturbance_map.grid_tensor - min_value + 1e-10
            )

        return radio_disturbance_map

    def get_3D_layer(self, val_array, height, radio_signal_height):  #
        delta_h = np.full(val_array.shape, height - radio_signal_height)
        new_val_array = np.zeros(val_array.shape)
        new_val_array[delta_h <= 0] = val_array[delta_h <= 0] / (
            (delta_h[delta_h <= 0] - 1) ** 2
        )
        new_val_array[delta_h > 0] = val_array[delta_h > 0] / (
            (delta_h[delta_h > 0] + 1) ** 2
        )
        return new_val_array
