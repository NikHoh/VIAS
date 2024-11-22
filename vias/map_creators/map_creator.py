import abc
import copy as cp
from typing import Dict

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin

from vias.grid_map import GridMap
from vias.utils.helpers import ScenarioInfo


def get_distances_to_nonzero_cells(input_array: np.ndarray) -> np.ndarray:
    input_array = cp.deepcopy(input_array)
    input_array = input_array.astype(bool)
    input_array = np.logical_not(input_array)
    output_array = distance_transform_edt(input_array)
    return output_array

def thin_nonzero_cells(input_array: np.ndarray) -> np.ndarray:
    input_array = cp.deepcopy(input_array)
    output_array = thin(input_array)
    return output_array


class MapCreator(object, metaclass=abc.ABCMeta):
    """This is an abstract class (~interface) for any grid_map creator that is implemented in the framework. Every grid_map creator
        has to override the abstract methods of this interface in order to work properly in the framework."""

    def __init__(self, inputs: Dict, scenario_info: ScenarioInfo, parameters: Dict):
        self.inputs = inputs
        self.scenario_info = scenario_info
        self.parameters = parameters

    @abc.abstractmethod
    def create_map(self, map_name) -> GridMap:
        """Creates a grid_map. The inputs for the grid_map creation can be found in self.input.
        After the ceration the grid_map is returned.
        """
        raise NotImplementedError('Users must define create_map() to use this base class.')

    def _get_map_blueprint(self, map_name: str, dimension: int):
        return GridMap(self.scenario_info.convert_to_map_info(map_name), dimension)
