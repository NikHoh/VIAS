import copy as cp
import math

import numpy as np

from vias.constraint_checkers.constraint_checker import Constraint, ConstraintChecker
from vias.grid_graph import load_grid_graph
from vias.grid_map import GridMap
from vias.map_creators.map_creator import get_distances_to_nonzero_cells
from vias.path import Path
from vias.simulators.grid_based import GridBased


class ConstraintCheckerStaticObstacleCollision(ConstraintChecker, GridBased):
    def __init__(self, **kwargs):
        ConstraintChecker.__init__(self, **kwargs)  # do not rely on super().__init__(
        # **kwargs) here as kwargs get
        # lost in MRO
        GridBased.__init__(self, **kwargs)  # do not rely on super().__init__(**kwargs)
        # here as kwargs get lost in MRO
        self.punishment_factor = kwargs["parameters"]["punishment_factor"]

    def load_grid_map(self) -> GridMap:
        return self.scenario.grid_maps[self.inputs[0]]

    def load_inputs(self, inputs_str: list[str], input_data_folder: str):
        for input_str in inputs_str:
            assert "map" in input_str, "Can not handle input other than GridMaps"
            self.inputs.append(input_str)
            if input_str not in self.scenario.grid_maps:
                grid_map = load_grid_graph(
                    input_data_folder, self.scenario.scenario_info, input_str
                )
                self.scenario.grid_maps[input_str] = grid_map

    def init(self):
        obstacle_grid_map = self.scenario.grid_maps[self.inputs[0]]
        inverse_obstacle_array = np.logical_not(obstacle_grid_map.grid_tensor)
        obstacle_distance_transform = get_distances_to_nonzero_cells(
            inverse_obstacle_array
        )
        self.processing["maximum_value"] = np.max(
            obstacle_distance_transform
        ) * np.size(obstacle_distance_transform)

        obstacle_distance_map = cp.deepcopy(obstacle_grid_map)
        obstacle_distance_map.clear()
        obstacle_distance_map.name = "obstacle_distance_map"
        obstacle_distance_map.set_from_array(obstacle_distance_transform)
        self.processing["obstacle_distance_map"] = obstacle_distance_map

    def check_constraint(self, path: Path) -> tuple[float, Constraint]:
        constraint_value = np.sum(
            self.processing[
                "obstacle_distance_map"
            ].get_interpolated_values_from_local_coords(path.waypoint_list)
        )

        if constraint_value > 0:
            punishment = self.punishment_factor * math.exp(
                constraint_value / self.processing["maximum_value"]
            )
        else:
            punishment = 0

        constraint = Constraint(
            "static_obstacle_collision", constraint_value, True, constraint_value <= 0
        )

        return punishment, constraint
