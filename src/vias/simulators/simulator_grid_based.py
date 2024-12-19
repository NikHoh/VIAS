import numpy as np

from vias.grid_graph import load_grid_graph
from vias.grid_map import GridMap
from vias.path import Path
from vias.simulators.grid_based import GridBased
from vias.simulators.simulator import Simulator


class SimulatorGridBased(Simulator, GridBased):
    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)  # do not rely on super().__init__(**kwargs)
        # here as kwargs get
        # lost in MRO
        GridBased.__init__(
            self, **kwargs
        )  # do not rely on super().__init__(**kwargs)  #
        # here as kwargs get  # lost in MRO

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

    def simulate(self, path: Path) -> float:
        grid_map = self.scenario.grid_maps[self.inputs[0]]

        # calculate trapezoidal integral
        distances = np.concatenate(
            (np.array([0]), np.sqrt(np.sum(np.diff(path.as_array(), axis=0) ** 2, 1)))
        )
        grid_values = grid_map.get_interpolated_values_from_local_coords(
            path.waypoint_list
        )
        simulator_result = np.trapz(grid_values, x=np.cumsum(distances))

        return simulator_result
