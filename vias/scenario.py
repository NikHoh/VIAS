from typing import Optional, Dict

from vias.grid_map import GridMap
from vias.utils.helpers import ScenarioInfo
from vias.console_manager import console


class Scenario:
    """This class calculates or generates information about the setting and the circumstances in that the traffic
    network optimization/ simulation will take place. This includes a 3D model of the city (not implemented yet), a
    corresponding set of 2d maps (grid maps), that contain static information about the vertiport positions
    (two_dim_map), no-fly-areas (nfa-grid_map), the risk
    to fly over a specific tile of the grid (risk-grid_map) and how heavily noise will affect the environment over a
    specific tile of the grid (noise-grid_map). Moreover, the scenario class consists of a
    logistics demand (namely a set of delivery tasks to be executed on designated vertiport positions)."""
    _instance: Optional['Scenario'] = None
    # scenario_info: Optional[ScenarioInfo] = None
    # grid_maps: Optional[Dict[str, GridMap]] = None
    # city_model: Optional[CityModel] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Scenario, cls).__new__(cls)
            # cls._instance.scenario_info: Optional[ScenarioInfo] = None
            # cls._instance.grid_maps: Optional[List[GridMap]] = None
            # cls._instance.city_model: Optional[CityModel] = None
        return cls._instance

    def __init__(self, *args):
        # Only initialize if it hasn't been initialized yet
        if not hasattr(self, '_initialized'):
            self._initialized = True  # Prevent further initialization
            if isinstance(args[0], ScenarioInfo):
                self.scenario_info = args[0]
                assert self.scenario_info.min_flight_height % self.scenario_info.z_res == 0, "If min_flight_height is not multiple of z_res, this can lead to strange behavior"
                assert self.scenario_info.max_flight_height % self.scenario_info.z_res == 0, "If max_flight_height is not multiple of z_res, this can lead to strange behavior"
            else:
                self.scenario_info: Optional[ScenarioInfo] = None
            self.grid_maps: Dict[str, GridMap] = {}

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    @property
    def x_length(self):
        return self.scenario_info.x_length

    @property
    def y_length(self):
        return self.scenario_info.y_length

    @property
    def z_length(self):
        return self.scenario_info.z_length

    @property
    def x_res(self):
        return self.scenario_info.x_res

    @property
    def y_res(self):
        return self.scenario_info.y_res

    @property
    def z_res(self):
        return self.scenario_info.z_res

    @property
    def min_flight_height(self):
        return self.scenario_info.min_flight_height

    @property
    def max_flight_height(self):
        return self.scenario_info.max_flight_height


