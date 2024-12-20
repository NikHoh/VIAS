import json
import os
import sys
from dataclasses import asdict, astuple, dataclass

import numpy as np

from vias.console_manager import console
from vias.utils.tools import (
    calculate_needed_graph_storage,
    euclidian_distance,
    float_2_str,
)


class OutOfOperationSpace(Exception):
    """Raised when point is going to lie outside of operation space bounds."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class PathEndsOutOfBound(Exception):
    """Raised when the start or the end point of the path can not be preprocessed
    because its too close to the maximum z-value of the scenario"""

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


@dataclass
class Info:
    map_NW_origin_lon: float
    map_NW_origin_lat: float
    tmerc_proj_origin_lon: float
    tmerc_proj_origin_lat: float
    x_length: int
    y_length: int
    z_length: int
    x_res: int
    y_res: int
    z_res: int

    def __post_init__(self):
        assert self.x_length % self.x_res == 0, (
            "x-length should be dividable by " "x-resolution."
        )
        assert self.y_length % self.y_res == 0, (
            "y-length should be dividable by " "y-resolution."
        )
        assert self.z_length % self.z_res == 0, (
            "z-length should be dividable by " "z-resolution."
        )


@dataclass
class ScenarioInfo(Info):
    user_identifier: str
    min_flight_height: int
    max_flight_height: int

    def __post_init__(self):
        assert self.x_length % 2 == 0, "Please use scenario x_length dividable by 2"
        assert self.y_length % 2 == 0, "Please use scenario y_length dividable by 2"
        assert self.x_length % self.x_res == 0, (
            "Please use scenario x_length that "
            "is properly dividable by scenario"
            " x_res"
        )
        assert self.y_length % self.y_res == 0, (
            "Please use scenario y_length that "
            "is properly dividable by scenario"
            " y_res"
        )
        assert self.z_length % self.z_res == 0, (
            "Please use scenario z_length that "
            "is properly dividable by scenario"
            " z_res"
        )
        assert self.max_flight_height % self.z_res == 0, (
            "Please use scenario "
            "max_flight_height that "
            "is properly dividable by "
            "scenario z_res"
        )
        assert self.min_flight_height % self.z_res == 0, (
            "Please use scenario "
            "min_flight_height that "
            "is properly dividable by "
            "scenario z_res"
        )
        columns = int(self.x_length / self.x_res)
        rows = int(self.y_length / self.y_res)
        layers = int(self.z_length / self.z_res)
        needed_storage = calculate_needed_graph_storage(columns, layers, rows)
        if needed_storage > 750:
            console.log(
                f"Needed storage to save a respective graph (directed, non clipped) "
                f"is. {needed_storage} MB.",
                highlight=True,
            )
            raise AssertionError("Grid too big")

    def convert_to_map_info(self, map_name: str) -> "MapInfo":
        return MapInfo(
            map_NW_origin_lon=self.map_NW_origin_lon,
            map_NW_origin_lat=self.map_NW_origin_lat,
            tmerc_proj_origin_lon=self.tmerc_proj_origin_lon,
            tmerc_proj_origin_lat=self.tmerc_proj_origin_lat,
            x_length=self.x_length,
            y_length=self.y_length,
            z_length=self.z_length,
            x_res=self.x_res,
            y_res=self.y_res,
            z_res=self.z_res,
            map_name=map_name,
        )


@dataclass
class MapInfo(Info):
    map_name: str

    def convert_to_scenario_info(
        self, user_identifier: str, min_flight_height: int, max_flight_height: int
    ) -> ScenarioInfo:
        return ScenarioInfo(
            map_NW_origin_lon=self.map_NW_origin_lon,
            map_NW_origin_lat=self.map_NW_origin_lat,
            tmerc_proj_origin_lon=self.tmerc_proj_origin_lon,
            tmerc_proj_origin_lat=self.tmerc_proj_origin_lat,
            x_length=self.x_length,
            y_length=self.y_length,
            z_length=self.z_length,
            x_res=self.x_res,
            y_res=self.y_res,
            z_res=self.z_res,
            user_identifier=user_identifier,
            min_flight_height=min_flight_height,
            max_flight_height=max_flight_height,
        )


# Save dataclass to JSON
def save_scenario_info_to_json(obj, path_folder):
    with open(os.path.join(path_folder, "scenario_info.json"), "w") as f:
        json.dump(asdict(obj), f)


# Load dataclass from JSON
def load_scenario_info_from_json(cls, path_folder):
    scenario_info_path = os.path.join(path_folder, "scenario_info.json")
    assert_msg = f"No scenario_info.json found at {scenario_info_path}"
    assert os.path.exists(scenario_info_path), assert_msg
    with open(scenario_info_path) as f:
        data = json.load(f)
    return cls(**data)


@dataclass
class ArrayCoord:
    row: int
    col: int
    lay: int

    def __post_init__(self):
        if self.row < 0:
            raise OutOfOperationSpace(
                f"ArrayCoord({self})", "Only positive row index allowed"
            )
        if self.col < 0:
            raise OutOfOperationSpace(
                f"ArrayCoord({self})", "Only positive column index allowed"
            )
        if self.lay < 0:
            raise OutOfOperationSpace(
                f"ArrayCoord({self})", "Only positive layer index allowed"
            )
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
class TmercCoord:
    east: float
    north: float

    def __post_init__(self):
        assert isinstance(self.east, float)
        assert isinstance(self.north, float)


@dataclass
class GlobalCoord:
    lon: float
    lat: float

    def __post_init__(self):
        assert isinstance(self.lon, float)
        assert isinstance(self.lat, float)


def get_osm_identifier(scenario_info: ScenarioInfo) -> str:
    """

    :param scenario_info:
    :return:
    """

    return (
        f"{scenario_info.user_identifier}_oLon_"
        f"{float_2_str(scenario_info.map_NW_origin_lon)}_"
        f"oLat_{float_2_str(scenario_info.map_NW_origin_lat)}_"
        f"x_{scenario_info.x_length}_y_{scenario_info.y_length}"
    )


def get_path_identifier(path_start: GlobalCoord, path_goal: GlobalCoord) -> str:
    return (
        f"_sLon_{float_2_str(path_start.lon)}_"
        f"sLat_{float_2_str(path_start.lat)}_"
        f"gLon_{float_2_str(path_goal.lon)}_"
        f"gLat{float_2_str(path_goal.lat)}"
    )


def get_map_identifier(scenario_info, dim: int):
    if dim == 2:
        return (
            f"{get_osm_identifier(scenario_info)}_"
            f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}"
        )
    elif dim == 3:
        return (
            f"{get_osm_identifier(scenario_info)}_"
            f"z_{scenario_info.z_length}_"
            f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}_resZ_"
            f"{scenario_info.z_res}"
        )
    else:
        raise Exception("vias.py", "Dimension not known")


def get_graph_identifier(scenario_info):
    return (
        f"{get_osm_identifier(scenario_info)}_"
        f"z_{scenario_info.z_length}_"
        f"resX_{scenario_info.x_res}_resY_{scenario_info.y_res}_resZ_"
        f"{scenario_info.z_res}"
    )


def get_mopp_identifier(
    scenario_info: ScenarioInfo, path_start: GlobalCoord, path_goal: GlobalCoord
) -> str:
    return (
        f"{get_map_identifier(scenario_info, 3)}"
        f"{get_path_identifier(path_start, path_goal)}"
    )


def is_stdout_redirected():
    return sys.stdout is not sys.__stdout__


def coastline2polygon(coastlines_dic: dict, x_length, y_length):
    if len(coastlines_dic) == 0:
        return [], []
    coastlines = [val["npos"] for val in coastlines_dic.values()]
    FUSE_DISTANCE = 5
    stacked_coastlines = []
    while len(coastlines) > 0:
        stacked_coastline = coastlines.pop(0)
        while True:
            if len(coastlines) == 0:
                break
            # distances from end of line to all other lines
            distances_to_start = [
                euclidian_distance(stacked_coastline[-1], line[0])
                for line in coastlines
            ]
            # distances from beginning of line to all other lines
            distances_to_end = [
                euclidian_distance(stacked_coastline[0], line[-1])
                for line in coastlines
            ]
            min_dist_to_start = min(distances_to_start)
            min_dist_to_end = min(distances_to_end)
            if min_dist_to_start < min_dist_to_end:
                if min_dist_to_start < FUSE_DISTANCE:
                    idx = distances_to_start.index(min_dist_to_start)
                    stacked_coastline.extend(coastlines.pop(idx))
                else:
                    break
            else:
                if min_dist_to_end < FUSE_DISTANCE:
                    idx = distances_to_end.index(min_dist_to_end)
                    stacked_coastline = coastlines.pop(idx) + stacked_coastline
                else:
                    break

        stacked_coastlines.append(stacked_coastline)

    # we have now all different coastlines stacked together
    # we assume that there is one large main coastline, all other coastlines are
    # assumed to be separate polygons (e.g. islands)
    coastline_length = [len(set(cst)) for cst in stacked_coastlines]
    max_idx = coastline_length.index(max(coastline_length))
    main_coastline = stacked_coastlines.pop(max_idx)

    # there are several cases where there has to be added points to the coastline
    start, goal = get_coastline_edges(main_coastline, x_length, y_length, 250)
    start_x = main_coastline[0][0]
    start_y = main_coastline[0][1]
    goal_x = main_coastline[-1][0]
    goal_y = main_coastline[-1][1]
    if start == "l" and goal == "l":
        if start_y > goal_y:
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
    if start == "b" and goal == "b":
        if start_x < goal_x:
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
    if start == "t" and goal == "t":
        if start_x > goal_x:
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
    if start == "r" and goal == "r":
        if start_y < goal_y:
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
    if start == "l" and goal == "t":
        main_coastline.append((0, y_length))
    if start == "l" and goal == "r":
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
    if start == "l" and goal == "b":
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
    if start == "b" and goal == "l":
        main_coastline.append((0, 0))
    if start == "b" and goal == "t":
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
    if start == "b" and goal == "r":
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
    if start == "r" and goal == "t":
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
    if start == "r" and goal == "l":
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
    if start == "r" and goal == "b":
        main_coastline.append((x_length, 0))
    if start == "t" and goal == "l":
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
    if start == "t" and goal == "b":
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
    if start == "t" and goal == "r":
        main_coastline.append((x_length, y_length))

    return main_coastline, stacked_coastlines


def get_coastline_edges(coastline, x_length, y_length, threshold):
    """Returns a combination of "l", "r", "t", "b" (left, right, top, bottom) depending
    on the position of the first and the last point of the coastline dependent to the
    frame."""
    ret = []
    for p in [coastline[0], coastline[-1]]:
        if p[0] < 0 + threshold:
            ret.append("l")
        elif p[0] > x_length - threshold:
            ret.append("r")
        elif p[1] < 0 + threshold:
            ret.append("b")
        elif p[1] > y_length - threshold:
            ret.append("t")
        else:
            ret.append(-1)
    return tuple(ret)
