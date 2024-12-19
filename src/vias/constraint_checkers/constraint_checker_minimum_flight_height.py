import math

import numpy as np

# from constraint_checker import ConstraintChecker
from vias.constraint_checkers.constraint_checker import Constraint, ConstraintChecker
from vias.grid_graph import load_grid_graph
from vias.grid_map import GridMap
from vias.path import Path
from vias.path_factory import PathFactory
from vias.simulators.grid_based import GridBased


class ConstraintCheckerMinimumFlightHeight(ConstraintChecker, GridBased):
    def __init__(self, **kwargs):
        ConstraintChecker.__init__(self, **kwargs)  # do not rely on super().__init__(
        # **kwargs) here as kwargs get
        # lost in MRO
        GridBased.__init__(
            self, **kwargs
        )  # do not rely on super().__init__(**kwargs) here as kwargs get lost in MRO
        self.punishment_factor = kwargs["parameters"]["punishment_factor"]
        self.min_flight_height = self.scenario.scenario_info.min_flight_height

    def load_inputs(self, inputs_str: list[str], input_data_folder: str):
        for input_str in inputs_str:
            assert "map" in input_str, "Can not handle input other than GridMaps"
            self.inputs.append(input_str)
            if input_str not in self.scenario.grid_maps:
                grid_map = load_grid_graph(
                    input_data_folder, self.scenario.scenario_info, input_str
                )
                self.scenario.grid_maps[input_str] = grid_map

    def load_grid_map(self) -> GridMap:
        return self.scenario.grid_maps[self.inputs[0]]

    def init(self):
        pass

    def check_constraint(self, path: Path) -> tuple[float, Constraint]:
        # cannot be calculated in init as num of control
        # points can change in preprocessor
        num_variable_control_points = PathFactory().num_variable_control_points
        maximum_value = num_variable_control_points * self.min_flight_height
        variable_control_points = path.nurbs_curve.ctrlpts[1:-1]
        cp_z = np.array([cp[2] for cp in variable_control_points])
        g = cp_z - np.ones(cp_z.shape) * self.min_flight_height  # g is array
        g[np.abs(g) < 1e-3] = 0
        constraint_value = sum(np.sqrt(np.heaviside(-g, 0) * g**2))

        if constraint_value > 0:
            punishment = self.punishment_factor * math.exp(
                constraint_value / maximum_value
            )
        else:
            punishment = 0

        constraint = Constraint(
            "minimum_flight_height", constraint_value, False, constraint_value <= 0
        )

        return punishment, constraint
