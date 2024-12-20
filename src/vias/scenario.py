from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vias.grid_map import GridMap

from pyproj import Proj

from vias.utils.helpers import ScenarioInfo, GlobalCoord, TmercCoord, Info, MapInfo


class Scenario:
    """This class calculates or generates information about the setting and the
    circumstances in that the traffic
    network optimization/ simulation will take place. This includes a 3D model of the
    city (not implemented yet), a
    corresponding set of 2d maps (grid maps), that contain static information about
    the vertiport positions
    (two_dim_map), no-fly-areas (nfa-grid_map), the risk
    to fly over a specific tile of the grid (risk-grid_map) and how heavily noise
    will affect the environment over a
    specific tile of the grid (noise-grid_map). Moreover, the scenario class consists
    of a
    logistics demand (namely a set of delivery tasks to be executed on designated
    vertiport positions)."""

    _instance: Optional["Scenario"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args):
        # Only initialize if it hasn't been initialized yet
        if not hasattr(self, "_initialized"):
            self._initialized = True  # Prevent further initialization
            if isinstance(args[0], ScenarioInfo):
                self.scenario_info = args[0]
                assert (
                    self.scenario_info.min_flight_height % self.scenario_info.z_res == 0
                ), (
                    "If min_flight_height is not "
                    "multiple of z_res, this can"
                    " lead to strange behavior"
                )
                assert (
                    self.scenario_info.max_flight_height % self.scenario_info.z_res == 0
                ), (
                    "If max_flight_height is not "
                    "multiple of z_res, this can"
                    " lead to strange behavior"
                )
            else:
                self.scenario_info: ScenarioInfo | None = None
            self.grid_maps: dict[str, "GridMap"] = {}

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def is_initialized(self):
        return hasattr(self, "_initialized") and self._initialized

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

    @property
    def map_NW_origin_lon(self):
        return self.scenario_info.map_NW_origin_lon

    @property
    def map_NW_origin_lat(self):
        return  self.scenario_info.map_NW_origin_lat

    @property
    def tmerc_proj_origin_lon(self):
        return self.scenario_info.tmerc_proj_origin_lon

    @property
    def tmerc_proj_origin_lat(self):
        return self.scenario_info.tmerc_proj_origin_lat


def get_projection():
    # EXAMPLE to use it
    # east_tmerc, north_tmerc = tmerc_projection(origin_lon, origin_lat)
    # lon_tmerc, lat_tmerc = tmerc_projection(diff_east, diff_south, inverse=True)
    scenario = Scenario()
    if scenario.is_initialized():
        lon_0 = scenario.tmerc_proj_origin_lon
        lat_0 = scenario.tmerc_proj_origin_lat
    else:
        lon_0 = 0.0
        lat_0 = 0.0
    tmerc_projection = Proj(proj="tmerc", ellps="WGS84", units="m",
                            lon_0=lon_0,
                            lat_0=lat_0)
    return tmerc_projection


def tmerc_coord_from_global_coord(
    global_coord: GlobalCoord, tmerc_projection: Proj | None = None
) -> TmercCoord:
    if tmerc_projection is None:
        tmerc_projection = get_projection()
    east, north = tmerc_projection(global_coord.lon, global_coord.lat)
    return TmercCoord(east, north)


def get_tmerc_map_center(info: Info):
    tmerc_map_origin = get_tmerc_map_origin(info)
    map_center_east = tmerc_map_origin.east + int(info.x_length / 2)
    map_center_north = tmerc_map_origin.north - int(info.y_length / 2)
    return TmercCoord(map_center_east, map_center_north)


def get_tmerc_map_origin(info: Info, tmerc_projection: Proj | None = None):
    if isinstance(info, ScenarioInfo | MapInfo):
        return tmerc_coord_from_global_coord(
            GlobalCoord(info.map_NW_origin_lon, info.map_NW_origin_lat),
            tmerc_projection=tmerc_projection,
        )
