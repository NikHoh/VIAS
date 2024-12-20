from typing import Optional

import numpy as np
from dotmap import DotMap
from geomdl import knotvector

from vias.config import get_config
from vias.console_manager import console
from vias.grid_graph import load_grid_graph, local_coord_from_tmerc_coord
from vias.path import ChordalNURBS, Path
from vias.scenario import Scenario, tmerc_coord_from_global_coord
from vias.utils.helpers import (
    ArrayCoord,
    GlobalCoord,
    LocalCoord,
)
from vias.utils.tools import get_equally_points_between


class PathFactory:
    _instance: Optional["PathFactory"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        # Only initialize if it hasn't been initialized yet
        if not hasattr(self, "_initialized"):
            self._initialized = True  # Prevent further initialization
            self.input_data_folder: str | None = None
            self.global_start_pos: GlobalCoord | None = None
            self.global_goal_pos: GlobalCoord | None = None
            self.local_start_pos: LocalCoord | None = None
            self.local_goal_pos: LocalCoord | None = None
            self.local_takeoff_pos: LocalCoord | None = None
            self.local_landing_pos: LocalCoord | None = None
            self.takeoff_sequence: list[LocalCoord] | None = None
            self.landing_sequence: list[LocalCoord] | None = None
            self.array_start_pos: ArrayCoord | None = None
            self.array_goal_pos: ArrayCoord | None = None

            config = get_config()

            expected_keyword_arguments = [
                "input_data_folder",
                "global_start_pos",
                "global_goal_pos",
            ]
            expected_data_types = [str, GlobalCoord, GlobalCoord]
            for idx, kwarg in enumerate(expected_keyword_arguments):
                if kwarg in kwargs:
                    setattr(self, kwarg, kwargs[kwarg])
                    assert isinstance(kwargs[kwarg], expected_data_types[idx]), (
                        f"Expected type {expected_data_types[idx]} but got "
                        f"{type(kwarg)}"
                    )
                else:
                    raise AssertionError(
                        f"Missing keyword argument {kwarg} upon first creation of "
                        f"PathFactory"
                    )

            # load needed grid maps into scenario
            scenario = Scenario()
            for input_str in ["buildings_map", "nfa_map"]:
                if input_str not in scenario.grid_maps:
                    grid_map = load_grid_graph(
                        self.input_data_folder, scenario.scenario_info, input_str
                    )
                    scenario.grid_maps[input_str] = grid_map

            # check for waypoint resolution
            # self.waypoint_resolution = config.path.waypoint_resolution
            self.waypoint_resolution = 0.5 * min(
                [
                    scenario.scenario_info.x_res,
                    scenario.scenario_info.y_res,
                    scenario.scenario_info.z_res,
                ]
            )
            console.log(
                f"The equidistant waypoint resolution is automatically determined to "
                f"be {self.waypoint_resolution} dependent on the scenario resolution."
            )
            assert self.waypoint_resolution <= min(
                [
                    scenario.scenario_info.x_res,
                    scenario.scenario_info.y_res,
                    scenario.scenario_info.z_res,
                ]
            ), (
                f"Waypoint resolution given in "
                f"the config folder is {self.waypoint_resolution} but "
                f"should be not bigger than "
                f"the smallest grid resolution"
            )

            if (
                isinstance(config.path.num_control_points, DotMap)
                and config.path.num_control_points.empty()
            ):
                assert "num_control_points" in kwargs, (
                    "Since the parameter "
                    "num_control_point was not "
                    "provided within the config "
                    "the PathFactory expects it "
                    "as keyword argument"
                )
                self.num_control_points = kwargs["num_control_points"]
            else:
                self.num_control_points = config.path.num_control_points

            if (
                isinstance(config.path.nurbs_order, DotMap)
                and config.path.nurbs_order.empty()
            ):
                assert "nurbs_order" in kwargs, (
                    "Since the parameter nurbs_order "
                    "was not provided within the "
                    "config the PathFactory expects it "
                    "as keyword argument"
                )
                self.nurbs_order = kwargs["nurbs_order"]
            else:
                self.nurbs_order = config.path.nurbs_order

            # get map_info for transformations
            map_info = scenario.grid_maps["buildings_map"].map_info

            # convert global to tmerc
            tmerc_start_pos = tmerc_coord_from_global_coord(self.global_start_pos)
            tmerc_goal_pos = tmerc_coord_from_global_coord(self.global_goal_pos)

            # convert tmerc to local
            local_start_pos_xy = local_coord_from_tmerc_coord(tmerc_start_pos, map_info)
            local_goal_pos_xy = local_coord_from_tmerc_coord(tmerc_goal_pos, map_info)

            # obtain takeoff height
            takeoff_height = scenario.grid_maps[
                "buildings_map"
            ].get_value_from_local_coord(local_start_pos_xy)
            # obtain landing height
            landing_height = scenario.grid_maps[
                "buildings_map"
            ].get_value_from_local_coord(local_goal_pos_xy)
            assert takeoff_height >= 0, (
                f"Takeoff height derived from buildings map "
                f"should be greater than zero but is {takeoff_height}."
            )
            assert landing_height >= 0, (
                f"Landing height derived from buildings map "
                f"should be greater than zero but is {landing_height}."
            )

            self.local_takeoff_pos = LocalCoord(
                local_start_pos_xy.x, local_start_pos_xy.y, takeoff_height
            )
            self.local_landing_pos = LocalCoord(
                local_goal_pos_xy.x, local_goal_pos_xy.y, landing_height
            )

            # obtain path start height
            array_takeoff_pos = scenario.grid_maps[
                "buildings_map"
            ].array_coord_from_local_coord(self.local_takeoff_pos)
            array_landing_pos = scenario.grid_maps[
                "buildings_map"
            ].array_coord_from_local_coord(self.local_landing_pos)

            start_lay = np.argmin(
                scenario.grid_maps["nfa_map"].grid_tensor[
                    array_takeoff_pos.row, array_takeoff_pos.col, :
                ]
            ).item()
            goal_lay = np.argmin(
                scenario.grid_maps["nfa_map"].grid_tensor[
                    array_landing_pos.row, array_landing_pos.col, :
                ]
            ).item()
            assert 0 <= start_lay < int(scenario.z_length / scenario.z_res), (
                f"Start of path layer index "
                f"{start_lay} lies out of "
                f"bound. Is it on a building "
                f"higher than the scenario's "
                f"z_length?"
            )
            assert 0 <= goal_lay < int(scenario.z_length / scenario.z_res), (
                f"Goal of path layer index "
                f"{goal_lay} lies out of "
                f"bound. Is it on a building "
                f"higher than the scenario's "
                f"z_length?"
            )

            self.array_start_pos = ArrayCoord(
                array_takeoff_pos.row, array_takeoff_pos.col, start_lay
            )
            self.array_goal_pos = ArrayCoord(
                array_landing_pos.row, array_landing_pos.col, goal_lay
            )

            self.local_start_pos = scenario.grid_maps[
                "nfa_map"
            ].local_coord_from_array_coord(self.array_start_pos)
            self.local_goal_pos = scenario.grid_maps[
                "nfa_map"
            ].local_coord_from_array_coord(self.array_goal_pos)

            # calculate takeoff sequence
            self.takeoff_sequence = [
                LocalCoord(*array.astype(float))
                for array in get_equally_points_between(
                    self.local_takeoff_pos.as_array(),
                    self.local_start_pos.as_array(),
                    self.waypoint_resolution,
                )
            ]
            # calculate landing sequence
            self.landing_sequence = [
                LocalCoord(*array.astype(float))
                for array in get_equally_points_between(
                    self.local_goal_pos.as_array(),
                    self.local_landing_pos.as_array(),
                    self.waypoint_resolution,
                )
            ]

            self.num_fixed_control_points = 2

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    @property
    def num_variable_control_points(self):
        return self.num_control_points - self.num_fixed_control_points

    @property
    def num_control_points(self):
        return self._num_control_points

    @num_control_points.setter
    def num_control_points(self, num_control_points):
        self._num_control_points = num_control_points

    @property
    def nurbs_order(self):
        return self._nurbs_order

    @nurbs_order.setter
    def nurbs_order(self, nurbs_order):
        self._nurbs_order = nurbs_order

    def generate_path_from_variable_control_points(
        self, control_points: list[LocalCoord], weight_vec: np.ndarray
    ) -> Path:
        # insert fixed control points
        assert self.local_start_pos is not None, "local_start_pos must be assigned"
        assert self.local_goal_pos is not None, "local_goal_pos must be assigned"
        control_points.insert(0, self.local_start_pos)
        control_points.append(self.local_goal_pos)

        # insert weights of fixed control points
        weights = np.hstack((np.ones((1,)), weight_vec, np.ones((1,))))

        nurbs_curve = ChordalNURBS(self.waypoint_resolution)
        nurbs_curve.degree = self.nurbs_order - 1

        # assign curve info
        nurbs_curve.ctrlpts = [
            control_points[i].as_array()
            for i in range(0, len(weights))
            if weights[i] != 0
        ]  # throw out control points with weight of zero
        nurbs_curve.weights = [
            weights[i] for i in range(0, len(weights)) if weights[i] != 0
        ]
        nurbs_curve.knotvector = knotvector.generate(
            self.nurbs_order - 1, len(nurbs_curve.ctrlpts)
        )

        path = Path(nurbs_curve)

        return path
