# from __future__ import annotations
from dataclasses import dataclass, astuple, asdict
from typing import Optional
from typing import List
import json
import sys
import os

import numpy as np
import plotly.io
from matplotlib import pyplot as plt
from pyproj import Proj
from scipy.interpolate import CloughTocher2DInterpolator as CT

from vias.config import get_config
from vias.utils.tools import float_2_str
from vias.console_manager import console



class OutOfOperationSpace(Exception):
    """Raised when point is going to lie outside of operation space bounds."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


@dataclass
class LocalCoord:
    x: float
    y: float
    z: float

    def __post_init__(self):
        assert isinstance(self.x, float)
        assert isinstance(self.y, float)
        assert isinstance(self.z, float)

    def as_array(self):
        return np.array(astuple(self))

    def as_homogeneous(self):
        components = list(astuple(self))
        components.append(1)  # append homogeneous 1
        return np.array(components)


@dataclass
class ArrayCoord:
    row: int
    col: int
    lay: int

    def __post_init__(self):
        if self.row < 0:
            raise OutOfOperationSpace(f"ArrayCoord({self})", "Only positive row index allowed")
        if self.col < 0:
            raise OutOfOperationSpace(f"ArrayCoord({self})", "Only positive column index allowed")
        if self.lay < 0:
            raise OutOfOperationSpace(f"ArrayCoord({self})", "Only positive layer index allowed")
        assert isinstance(self.row, int)
        assert isinstance(self.col, int)
        assert isinstance(self.lay, int)

    def as_array(self):
        return np.array(astuple(self))

    def as_homogeneous(self):
        components = list(astuple(self))
        components.append(1)  # append homogeneous 1
        return np.array(components)


@dataclass
class Info:
    map_NW_origin_lon: float
    map_NW_origin_lat: float
    x_length: int
    y_length: int
    z_length: int
    x_res: int
    y_res: int
    z_res: int

    def __post_init__(self):
        assert self.x_length % self.x_res == 0, "x-length should be dividable by x-resolution."
        assert self.y_length % self.y_res == 0, "y-length should be dividable by y-resolution."
        assert self.z_length % self.z_res == 0, "z-length should be dividable by z-resolution."


@dataclass
class ScenarioInfo(Info):
    user_identifier: str
    min_flight_height: int
    max_flight_height: int

    def __post_init__(self):
        assert self.x_length % 2 == 0, "Please use scenario x_length dividable by 2"
        assert self.y_length % 2 == 0, "Please use scenario y_length dividable by 2"
        assert self.x_length % self.x_res == 0, "Please use scenario x_length that is properly dividable by scenario x_res"
        assert self.y_length % self.y_res == 0, "Please use scenario y_length that is properly dividable by scenario y_res"
        assert self.z_length % self.z_res == 0, "Please use scenario z_length that is properly dividable by scenario z_res"
        assert self.max_flight_height % self.z_res == 0, "Please use scenario max_flight_height that is properly dividable by scenario z_res"
        assert self.min_flight_height % self.z_res == 0, "Please use scenario min_flight_height that is properly dividable by scenario z_res"
        columns = int(self.x_length/self.x_res)
        rows = int(self.y_length/self.y_res)
        layers = int(self.z_length/self.z_res)
        needed_storage = calculate_needed_graph_storage(columns, layers, rows)
        if needed_storage > 750:
            console.log(f"Needed storage to save a respective graph (directed, non clipped) is. {needed_storage} MB.", highlight=True)

    def convert_to_map_info(self, map_name: str) -> "MapInfo":
        return MapInfo(map_NW_origin_lon=self.map_NW_origin_lon,
                       map_NW_origin_lat=self.map_NW_origin_lat,
                       x_length=self.x_length,
                       y_length=self.y_length,
                       z_length=self.z_length,
                       x_res=self.x_res,
                       y_res=self.y_res,
                       z_res=self.z_res,
                       map_name=map_name)

# Save dataclass to JSON
def save_scenario_info_to_json(obj, path_folder):
    with open(os.path.join(path_folder, "scenario_info.json"), 'w') as f:
        json.dump(asdict(obj), f)

# Load dataclass from JSON
def load_scneario_info_from_json(cls, path_folder):
    scenario_info_path = os.path.join(path_folder, "scenario_info.json")
    if not os.path.exists(scenario_info_path):
        return None
    with open(scenario_info_path, 'r') as f:
        data = json.load(f)
    return cls(**data)


@dataclass
class MapInfo(Info):
    map_name: str

    def convert_to_scenario_info(self, user_identifier: str, min_flight_height: int,
                                 max_flight_height: int) -> ScenarioInfo:
        return ScenarioInfo(map_NW_origin_lon=self.map_NW_origin_lon,
                            map_NW_origin_lat=self.map_NW_origin_lat,
                            x_length=self.x_length,
                            y_length=self.y_length,
                            z_length=self.z_length,
                            x_res=self.x_res,
                            y_res=self.y_res,
                            z_res=self.z_res,
                            user_identifier=user_identifier,
                            min_flight_height=min_flight_height,
                            max_flight_height=max_flight_height)


@dataclass
class GlobalCoord:
    lon: float
    lat: float

    def __post_init__(self):
        assert isinstance(self.lon, float)
        assert isinstance(self.lat, float)


@dataclass
class TmercCoord:
    east: float
    north: float

    def __post_init__(self):
        assert isinstance(self.east, float)
        assert isinstance(self.north, float)


# def get_projection():
#     return Proj(proj="tmerc",
#                 ellps="WGS84",
#                 units="m")

def get_projection():
    # EXAMPLE to use it
    # east_tmerc, north_tmerc = tmerc_projection(promis_origin_lon, promis_origin_lat)
    # lon_tmerc, lat_tmerc = tmerc_projection(diff_east, diff_south, inverse=True)
    promis_origin_lon = 2.339144258065529
    promis_origin_lat = 48.86776494028203
    tmerc_projection = Proj(
        proj="tmerc",
        ellps="WGS84",
        units="m",
        lon_0=promis_origin_lon,
        lat_0=promis_origin_lat)
    return tmerc_projection

def tmerc_coord_from_global_coord(global_coord: GlobalCoord, tmerc_projection: Optional[Proj] = None) -> TmercCoord:
    if tmerc_projection is None:
        tmerc_projection = get_projection()
    east, north = tmerc_projection(global_coord.lon, global_coord.lat)
    return TmercCoord(east, north)


def get_osm_identifier(scenario_info: ScenarioInfo) -> str:
    """

    :param scenario_info:
    :return:
    """

    return (f"{scenario_info.user_identifier}_oLon_{float_2_str(scenario_info.map_NW_origin_lon)}_"
            f"oLat_{float_2_str(scenario_info.map_NW_origin_lat)}_"
            f"x_{scenario_info.x_length}_y_{scenario_info.y_length}")


def get_map_identifier(scenario_info, dim: int):
    if dim == 2:
        return (f"{get_osm_identifier(scenario_info)}_"
                f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}")
    elif dim == 3:
        return (f"{get_osm_identifier(scenario_info)}_"
                f"z_{scenario_info.z_length}_"
                f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}_resZ_{scenario_info.z_res}")
    else:
        raise Exception("vias.py", "Dimension not known")

def get_graph_identifier(scenario_info):
    return (f"{get_osm_identifier(scenario_info)}_"
                f"z_{scenario_info.z_length}_"
                f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}_resZ_{scenario_info.z_res}")


def get_tmerc_map_center(info: Info):
    tmerc_map_origin = get_tmerc_map_origin(info)
    map_center_east = tmerc_map_origin.east + int(info.x_length / 2)
    map_center_north = tmerc_map_origin.north - int(info.y_length / 2)
    return TmercCoord(map_center_east, map_center_north)


def get_tmerc_map_origin(info: Info, tmerc_projection: Optional[Proj] = None):
    if isinstance(info, ScenarioInfo):
        return tmerc_coord_from_global_coord(GlobalCoord(info.map_NW_origin_lon, info.map_NW_origin_lat),
                                             tmerc_projection=tmerc_projection)
    elif isinstance(info, MapInfo):
        return tmerc_coord_from_global_coord(GlobalCoord(info.map_NW_origin_lon, info.map_NW_origin_lat),
                                             tmerc_projection=tmerc_projection)

def get_equally_points_between(p1: np.ndarray, p2: np.ndarray, max_distance: float) -> List[np.ndarray]:
    total_distance = np.linalg.norm(p2 - p1)

    # Determine the number of intermediate points
    num_points = int(np.ceil(total_distance / max_distance)) - 1

    # Generate equally spaced points along the line
    points = [p1 + (p2 - p1) * (i / (num_points + 1)) for i in range(1, num_points + 1)]

    return [p1] + points + [p2]

def get_specific_number_of_points_between(p1: np.ndarray, p2: np.ndarray, num_points: int) -> List[np.ndarray]:
    points = [p1 + (p2 - p1) * (i / (num_points + 1)) for i in range(1, num_points + 1)]
    assert len(points) == num_points
    return points

class PathEndsOutOfBound(Exception):
    "Raised when the start or the end point of the path can not be preprocessed because its too close to the maximum z-value of the scenario"
    def __init(self, expression, message):
        self.expression = expression
        self.message = message


class NoValidPathError(Exception):
    """Raised when no valid path is found."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class NoSuitableApproximation(Exception):
    """Raised when no NURBS approximation for a Dijkstra path is found."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class OptimizationEndsInNiches(Exception):
    """Raised when the optimization terminates while niches have not resolved yet."""
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def _save_and_close_plotly(close, fig, savepath, show_figure):
    config = get_config()
    # Save the plot if needed
    fig.update_layout(showlegend=False)
    fig.data[0].update(showscale=False)
    if savepath != '' and not config.suppress_grid_image_save:
        fig.write_image(savepath + '.png', height=2080, width=2080)
        # fig.write_html(savepath + '.html')
        # fig.write_json(savepath + '.json')
        if config.save_as_pdf:
            fig.write_image(savepath + '.pdf')  # Plotly supports SVG, but not EPS directly
    # Show the image if requested
    # if not config.suppress_grid_image_plot and show_figure:
        # fig.show()
    if close:
        fig = None
    return fig


def _save_and_close_matplotlib(close, savepath, show_figure):
    config = get_config()
    if savepath != '' and not config.suppress_grid_image_save:
        plt.savefig(''.join([savepath, '.png']), bbox_inches="tight")
    if config.save_as_pdf:
        plt.savefig(''.join([savepath, '.pdf']), bbox_inches="tight")
    if not config.suppress_grid_image_plot and show_figure:
        plt.show()
    if close:
        plt.close()


def get_num_nodes(columns, layers, rows):
    return rows * columns * layers


def calculate_needed_graph_storage(columns, layers, rows):
    needed_storage = (4 * get_num_nodes(columns, layers, rows) + 2 * get_num_edges(columns, layers, rows,
                                                                                   True)) * 8 / 1E6  # in MB
    return needed_storage


def get_num_edges(columns, layers, rows, directed):
    # assuming a bidirectional 26 neighborhood
    num_expected_edges = (
            ((layers - 2) * (rows - 2) * (columns - 2) * 26)  # inner nodes
            + ((layers - 2) * (rows - 2) * 2 + (layers - 2) * (columns - 2) * 2 + (rows - 2) * (
            columns - 2) * 2) * 17  # outer plane nodes
            + ((layers - 2) * 4 + (rows - 2) * 4 + (columns - 2) * 4) * 11  # edge nodes
            + 8 * 7)  # corner nodes
    if not directed:
        return int(num_expected_edges / 2)
    return num_expected_edges


def get_path_identifier(path_start: GlobalCoord, path_goal: GlobalCoord) -> str:
    return (f"_sLon_{float_2_str(path_start.lon)}_"
            f"sLat_{float_2_str(path_start.lat)}_"
            f"gLon_{float_2_str(path_goal.lon)}_"
            f"gLat{float_2_str(path_goal.lat)}")


def get_mopp_identifier(scenario_info: ScenarioInfo,
                        path_start: GlobalCoord,
                        path_goal: GlobalCoord) -> str:
    return (f"{get_map_identifier(scenario_info, 3)}"
            f"{get_path_identifier(path_start, path_goal)}")


def interpolation_with_extrapolation2D(xy, z):
    x = xy[:, 0]
    y = xy[:, 1]
    f = CT(xy, z)

    # this inner function will be returned to a user
    def new_f(xx, yy):
        # evaluate the CT interpolator. Out-of-bounds values are nan.
        zz = f(xx, yy)
        nans = np.isnan(zz)

        if nans.any():
            # for each nan point, find its nearest neighbor
            inds = np.argmin(
                (x[:, None] - xx[nans]) ** 2 +
                (y[:, None] - yy[nans]) ** 2
                , axis=0)
            # ... and use its value
            zz[nans] = z[inds]
        return zz

    return new_f


def is_stdout_redirected():
    return sys.stdout is not sys.__stdout__