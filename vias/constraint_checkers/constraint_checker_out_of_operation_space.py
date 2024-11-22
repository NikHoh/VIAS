import math
from abc import ABC
from typing import Tuple, List

import numpy as np

from vias.constraint_checkers.constraint_checker import Constraint, ConstraintChecker
from vias.path import Path
from vias.simulators.non_graph_based import NonGraphBased
from vias.utils.math_functions import EPS


class ConstraintCheckerOutOfOperationSpace(ConstraintChecker, NonGraphBased):
    def __init__(self, **kwargs):
        ConstraintChecker.__init__(self, **kwargs)  # do not rely on super().__init__(**kwargs) here as kwargs get lost in MRO
        NonGraphBased.__init__(self)  # do not rely on super().__init__(**kwargs) here as kwargs get lost in MRO
        self.punishment_factor = kwargs["parameters"]["punishment_factor"]

    def load_inputs(self, inputs_str: List[str], input_data_folder: str):
        pass

    def init(self):
        pass

    def check_constraint(self, path: Path) -> Tuple[float, Constraint]:

        maximum_value = 3 * 100.0 * len(
            path.waypoint_list)  # in all three dimensions all waypoints lie 100m out of bound

        # punish lower bounds
        lower_bound_array = path.as_array()[path.as_array() < 0-1E-3]
        constraint_value = abs(sum(lower_bound_array))
        # punish upper bounds
        upper_bound_array = np.full(path.as_array().shape,
                                    [self.scenario.x_length, self.scenario.y_length,
                                     self.scenario.z_length]) - path.as_array()
        constraint_value += abs(sum(upper_bound_array[upper_bound_array < 0-1E-3]))

        if constraint_value > 0:
            punishment = self.punishment_factor * math.exp(constraint_value / maximum_value)
        else:
            punishment = 0

        constraint = Constraint('path_out_of_operation_space', constraint_value, True, constraint_value <= 0)

        return punishment, constraint
