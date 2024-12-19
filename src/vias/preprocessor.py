import copy as cp
import itertools
import math
import os
import time
from dataclasses import astuple

import igraph
import igraph as ig
import numpy as np
import scipy.ndimage
from pymoo.util.ref_dirs import get_reference_directions
from scipy import stats

from vias.config import get_config
from vias.console_manager import console
from vias.data_manager import DataManager
from vias.evaluator import Evaluator
from vias.grid_graph import (
    GridGraph,
    get_grid_graph_path,
    load_grid_graph,
    save_grid_graph,
)
from vias.path import Path
from vias.path_factory import PathFactory
from vias.scenario import Scenario
from vias.utils.helpers import (
    ArrayCoord,
    LocalCoord,
    NoSuitableApproximation,
    get_path_identifier,
)
from vias.utils.math_functions import chamfer_distance
from vias.utils.tools import (
    bcolors as bc,
)
from vias.utils.tools import (
    calculate_needed_graph_storage,
    get_num_edges,
    get_num_nodes,
    get_specific_number_of_points_between,
)


def max_without_outliers_zscore(data, threshold=3):
    data = np.array(data)

    z_scores = np.abs(stats.zscore(data))

    filtered_data = data[z_scores < threshold]

    return np.max(filtered_data)


def normalize_graphs(
    grid_graphs: dict[str, GridGraph], normalization_type: str
) -> dict[str, GridGraph]:
    # do normalization
    if normalization_type == "LSTSQ":
        # first get min and max for all grids
        E = len(grid_graphs)
        same_vals_3D = np.zeros(E, dtype=int)
        minima = {}
        maxima = {}
        # find minima and maxima of every graph
        for idx, (objective_str, grid_graph) in enumerate(grid_graphs.items()):
            minimum_value = min(grid_graph.graph.es[objective_str])
            maximum_value = max(grid_graph.graph.es[objective_str])

            if minimum_value == maximum_value:
                minimum_value = math.inf
                maximum_value = math.inf
                same_vals_3D[idx] = 1

            minima[idx] = minimum_value
            maxima[idx] = maximum_value

        # solve A@x = b

        # do the least square normalization described in the diss
        E_adap = E - int(sum(same_vals_3D))
        num_eq = int(E_adap * (E_adap - 1) / 2)
        A = np.zeros((num_eq, E_adap))
        b = np.zeros(num_eq)
        idx_dict = dict(
            zip(
                np.array(range(0, E))[same_vals_3D is False],
                np.array(range(0, E_adap)),
                strict=False,
            )
        )
        for idx, (obj1, obj2) in enumerate(
            itertools.combinations(np.array(range(0, E))[same_vals_3D is False], 2)
        ):
            A[idx, idx_dict[obj1]] = maxima[obj1] - minima[obj1]
            A[idx, idx_dict[obj2]] = minima[obj2] - maxima[obj2]
        x_3D_final = np.ones(E)
        if A.size != 0:
            x_3D = scipy.optimize.lsq_linear(A, b, bounds=(1e-30, np.inf)).x
            x_3D_final[same_vals_3D is False] = x_3D

    normalized_graphs = {}
    for idx, (objective_str, grid_graph) in enumerate(grid_graphs.items()):
        weight_array = np.array(grid_graph.graph.es[objective_str])
        # normalize grid grid_map
        if normalization_type == "LSTSQ":
            grid_graph.graph.es[objective_str] = weight_array * x_3D_final[idx]
        elif normalization_type == "DBS":
            grid_graph.graph.es[objective_str] = weight_array / (weight_array.sum())
        elif normalization_type == "NONE":
            pass
        else:
            raise AssertionError("Normalization type not known")
        if not np.all(np.array(grid_graph.graph.es[objective_str]) > 0):
            assert 1 == 0, (
                "Dijkstra won't work with negative weights. Zero weights "
                "should be avoided, to avoid 'No-Cost-Paths'"
            )
        normalized_graphs[objective_str] = grid_graph
    return normalized_graphs


def clip_and_merge_graphs(
    grid_graphs: dict[str, GridGraph], normalization_type: str
) -> GridGraph:
    normalized_grid_graphs = normalize_graphs(grid_graphs, normalization_type)

    map_info = list(grid_graphs.values())[0].map_info
    dim = list(grid_graphs.values())[0].dimension

    # ig.union can only work with directed or undirected graphs, so if there is at
    # least one directed, convert all to directed
    if any([grid_graph.directed for grid_graph in normalized_grid_graphs.values()]):
        for grid_graph in normalized_grid_graphs.values():
            if not grid_graph.directed:
                grid_graph.graph = grid_graph.graph.as_directed(mode="mutual")

    union_graph = ig.union(
        [grid_graph.graph for grid_graph in normalized_grid_graphs.values()]
    )

    # union changes the order in the objectives, which may have strange effects,
    # so we build a new graph with correct order
    saved_attributes = {}
    for attribute in union_graph.edge_attributes():
        saved_attributes[attribute] = union_graph.es[attribute]
        del union_graph.es[attribute]

    for key in grid_graphs:
        union_graph.es[key] = saved_attributes[key]

    merged_graph = GridGraph(map_info, union_graph, dimension=dim)

    return merged_graph


def get_nfa_union_graph(nfa_graphs) -> igraph.Graph:
    nfa_union_graph = ig.union(
        [grid_graph.graph for grid_graph in nfa_graphs], byname=True
    )
    return nfa_union_graph


class PreProcessor:
    def __init__(self, evaluator):
        config = get_config()
        self.path_factory = PathFactory()
        self.evaluator: Evaluator = evaluator
        self.scenario = Scenario()
        rows = int(
            self.scenario.scenario_info.y_length / self.scenario.scenario_info.y_res
        )
        columns = int(
            self.scenario.scenario_info.x_length / self.scenario.scenario_info.x_res
        )
        layers = int(
            self.scenario.scenario_info.z_length / self.scenario.scenario_info.z_res
        )
        self.default_num_graph_edges = get_num_edges(columns, layers, rows, True)
        self.default_num_graph_nodes = get_num_nodes(columns, layers, rows)
        self.clipped_num_graph_edges: int | None = None
        self.clipped_num_graph_nodes: int | None = None

        needed_storage = calculate_needed_graph_storage(columns, layers, rows)
        if needed_storage > 750:
            console.log(
                f"Saving the worst case graph (directed not clipped) would roughly "
                f"take {needed_storage}MB of storage. Exiting.",
                highlight=True,
            )
            raise AssertionError("Graph too big")

        if (
            config.preprocessor.adaptive_number_of_control_points
            and not config.preprocessor.advanced_individual_init
        ):
            assert 1 == 0, (
                "Adaptive number of control points can only be used with "
                "advanced initialization"
            )

    def calculate_initial_paths(self) -> list[Path]:
        config = get_config()
        if not config.suppress_preprocessor_paths_plot:
            grid_map = load_grid_graph(
                DataManager().data_input_path,
                self.scenario.scenario_info,
                "buildings_map",
            )
        if config.preprocessor.advanced_individual_init:
            dijkstra_paths = self.calculate_advanced_individuals()
            if not config.suppress_preprocessor_paths_plot:
                suffix = "_{}".format(
                    get_path_identifier(
                        self.path_factory.global_start_pos,
                        self.path_factory.global_goal_pos,
                    )
                )
                grid_map.plot_paths(
                    dijkstra_paths,
                    savepath=DataManager().dijkstra_paths_path,
                    prefix="2D_dijkstra_paths_",
                    suffix=suffix,
                )
                suffix = "_{}".format(
                    get_path_identifier(
                        self.path_factory.global_start_pos,
                        self.path_factory.global_goal_pos,
                    )
                )
                grid_map.plot_3D_paths(
                    dijkstra_paths,
                    savepath=DataManager().dijkstra_paths_path,
                    prefix="3D_dijkstra_paths_",
                    suffix=suffix,
                )
            if config.preprocessor.adaptive_number_of_control_points:
                approximation_error_threshold = (
                    config.preprocessor.approximation_error_threshold
                )
                while True:
                    try:
                        console.log(
                            f"Trying ANCP run with approximation error "
                            f"threshold {approximation_error_threshold}."
                        )
                        new_num_cp, approximated_dijkstra_paths = (
                            self.get_optimal_num_cp(
                                dijkstra_paths, approximation_error_threshold
                            )
                        )
                        break
                    except NoSuitableApproximation:
                        console.log(
                            "No suitable approximation found. Increase error "
                            "threshold and try again."
                        )
                        approximation_error_threshold += 0.1
                # maybe the number of control points has adapted --> adapt coder
                self.path_factory.num_control_points = new_num_cp
                initial_paths = [
                    self.get_straight_path()
                ]  # # to also have straight path in there
                initial_paths.extend(approximated_dijkstra_paths)
            else:
                initial_paths = [
                    self.get_straight_path()
                ]  # # to also have straight path in there
                approximated_dijkstra_paths = cp.deepcopy(dijkstra_paths)
                for path in approximated_dijkstra_paths:
                    path.smooth(1)
                    path.approximate_nurbs(
                        self.path_factory.num_control_points,
                        self.path_factory.nurbs_order,
                        self.path_factory.waypoint_resolution,
                    )  # // TODO maybe  # shift this method  #  # into  # path_factory?
                initial_paths.extend(approximated_dijkstra_paths)

        else:  # initialize with straight path
            initial_paths = [self.get_straight_path()]
        if not config.suppress_preprocessor_paths_plot:
            suffix = "_{}".format(
                get_path_identifier(
                    self.path_factory.global_start_pos,
                    self.path_factory.global_goal_pos,
                )
            )
            grid_map.plot_3D_paths(
                initial_paths,
                savepath=DataManager().dijkstra_paths_path,
                prefix="3D_initial_paths_",
                suffix=suffix,
            )
        return initial_paths

    def get_optimal_num_cp(
        self, dijkstra_paths: list[Path], approximation_error_threshold: float
    ) -> tuple[int, list[Path]]:
        get_config()
        max_num_control_points = int(
            2.5
            * max(
                self.scenario.x_length, self.scenario.y_length, self.scenario.z_length
            )
            / min(self.scenario.x_res, self.scenario.y_res, self.scenario.z_res)
        )

        min_num_cp_for_approx = self.path_factory.nurbs_order

        # do smoothing to make it easier for spline approximation
        smoothed_dijkstra_paths = cp.deepcopy(dijkstra_paths)
        for path in smoothed_dijkstra_paths:
            path.smooth(1)

        grid_map = load_grid_graph(
            DataManager().data_input_path, self.scenario.scenario_info, "buildings_map"
        )
        suffix = "_{}".format(
            get_path_identifier(
                self.path_factory.global_start_pos, self.path_factory.global_goal_pos
            )
        )
        grid_map.plot_3D_paths(
            smoothed_dijkstra_paths,
            savepath=DataManager().dijkstra_paths_path,
            prefix="3D_smoothed_paths_",
            suffix=suffix,
        )

        optimum_number_of_cps = []

        for smoothed_dijkstra_path in smoothed_dijkstra_paths:
            cp_error_dic = {}
            filling_flag = False
            num_cp_for_approx = min_num_cp_for_approx
            while True:
                path = cp.deepcopy(smoothed_dijkstra_path)
                path.approximate_nurbs(
                    num_cp_for_approx,
                    self.path_factory.nurbs_order,
                    self.path_factory.waypoint_resolution,
                )

                error = chamfer_distance(
                    path.as_array(), smoothed_dijkstra_path.as_array(), direction="max"
                )

                cp_error_dic[num_cp_for_approx] = error / smoothed_dijkstra_path.length

                if (
                    error
                    <= approximation_error_threshold * smoothed_dijkstra_path.length
                ):
                    filling_flag = True
                    num_cp_for_approx -= 1
                    if num_cp_for_approx < min_num_cp_for_approx:
                        break
                    if num_cp_for_approx - 1 in cp_error_dic:
                        break

                else:
                    if filling_flag:
                        break
                    num_cp_for_approx += 5  # increase CPs in steps of 5
                if num_cp_for_approx > max_num_control_points:
                    raise NoSuitableApproximation(
                        "get_optimal_num_cp()",
                        "Maximum number of control points "
                        "reached "
                        "without meeting the desired error "
                        "threshold. Try increasing the "
                        "threshold "
                        "approximation_error_threshold and "
                        "rerun.",
                    )

            # sort dicts to ascending number of control points
            cp_error_dic = dict(sorted(cp_error_dic.items()))

            mask_cps = [
                num_cp
                for i, num_cp in enumerate(cp_error_dic.keys())
                if list(cp_error_dic.values())[i] < approximation_error_threshold
            ]

            # catch case where there is no suitable approximation
            if len(mask_cps) == 0:
                console.log(
                    bc.FAIL
                    + "Found no suitable approximation. Will end program."
                    + bc.ENDC
                )
                raise NoSuitableApproximation(
                    "get_optimal_num_cp", "Did not find suitable approximation"
                )

            cp_to_choose = mask_cps[0]
            optimum_number_of_cps.append(cp_to_choose)

            console.print(
                f"Best curve approximation is that with error of "
                f"{cp_error_dic[cp_to_choose]} for {cp_to_choose} control "
                f"points",
                end="\r",
            )

        # the optimal number of control points for different objectives can differ,
        # choose maximum
        ancp_num_cp = max_without_outliers_zscore(optimum_number_of_cps)
        console.log(f"Final chosen number of control points is {ancp_num_cp}.")
        for path in smoothed_dijkstra_paths:
            path.approximate_nurbs(
                ancp_num_cp,
                self.path_factory.nurbs_order,
                self.path_factory.waypoint_resolution,
            )

        return ancp_num_cp, smoothed_dijkstra_paths

    def get_straight_path(self):
        start_point = self.path_factory.local_start_pos
        goal_point = self.path_factory.local_goal_pos
        num_variable_control_points = self.path_factory.num_variable_control_points
        variable_control_points = get_specific_number_of_points_between(
            start_point.as_array(), goal_point.as_array(), num_variable_control_points
        )
        weights = np.ones(len(variable_control_points))
        path = self.path_factory.generate_path_from_variable_control_points(
            [LocalCoord(*point) for point in variable_control_points], weights
        )
        return path

    def calculate_advanced_individuals(self) -> list[Path]:
        nfa_graphs = []
        for constraint_checker in self.evaluator.constraint_checkers:
            nfa_graph = constraint_checker.derive_grid_graph()
            if nfa_graph is not None:
                nfa_graphs.append(nfa_graph)
        nfa_union_graph = get_nfa_union_graph(nfa_graphs)

        grid_graphs = {}
        for simulator in self.evaluator.simulators:
            grid_graph = simulator.derive_grid_graph(nfa_union_graph)
            if grid_graph is not None:
                grid_graphs[simulator.objective_str] = grid_graph

        if len(grid_graphs) == 0:
            return []

        num_grid_graphs = len(grid_graphs)

        merged_graph = self.get_merged_graph(grid_graphs)

        combinations = self.get_combinations(num_grid_graphs)

        console.log("Preprocessing module is working. This may take a while ... ")
        # get advanced control points for every grid_graph respectively

        # get optimal path consisting of points

        dijkstra_paths, _ = self.calculate_dijkstra_paths(merged_graph, combinations)

        return dijkstra_paths

    def get_combinations(self, num_grid_graphs):
        config = get_config()
        if config.preprocessor.multiple_weighted_start_points:
            combinations = get_reference_directions(
                "uniform",
                num_grid_graphs,
                n_points=config.preprocessor.num_mwsp,
                seed=config.seed,
            )
        else:
            combinations = get_reference_directions(
                "uniform", num_grid_graphs, n_points=num_grid_graphs, seed=config.seed
            )

        combinations[combinations == 0.0] = 1e-6
        return combinations

    def calculate_dijkstra_paths(
        self, merged_graph: GridGraph, combinations: np.ndarray
    ) -> tuple[list[Path], list[float]]:
        """Calculates an optimal set of points, that lead from start to goal in a 3D
        grid grid_map. Uses a bidirectional
        Dijkstra graph algorithm for that."""

        # let A* find the optimal path
        start_node = self.path_factory.array_start_pos
        goal_node = self.path_factory.array_goal_pos

        start_node_index = merged_graph.graph.vs["name"].index(astuple(start_node))
        goal_node_index = merged_graph.graph.vs["name"].index(astuple(goal_node))

        all_weights = np.zeros(
            (len(merged_graph.graph.es), len(merged_graph.graph.edge_attributes()))
        )
        cost_values = []
        dijkstra_paths = []

        for idx, objective_str in enumerate(merged_graph.graph.edge_attributes()):
            all_weights[:, idx] = merged_graph.graph.es[objective_str]

        list_of_vertex_id_paths = []

        number_iterations = len(combinations)
        for comb_idx, combination in enumerate(combinations):
            console.print(
                f"Dijkstra run {comb_idx + 1} of {number_iterations}", end="\r"
            )
            comb = np.array(combination)
            assert comb.shape[0] == all_weights.shape[1]
            weights_list = np.dot(all_weights, comb).flatten().tolist()
            vertex_id_path = merged_graph.graph.get_shortest_path(
                start_node_index,
                to=goal_node_index,
                weights=weights_list,
                output="vpath",
            )

            if vertex_id_path not in list_of_vertex_id_paths:
                list_of_vertex_id_paths.append(vertex_id_path)
                edge_id_path = np.array(
                    [
                        merged_graph.graph.get_eid(
                            vertex_id_path[i], vertex_id_path[i + 1]
                        )
                        for i in range(len(vertex_id_path) - 1)
                    ]
                )

                cost_value = np.sum(np.array(weights_list)[edge_id_path])

                cost_values.append(cost_value)
                array_coords = [
                    ArrayCoord(*merged_graph.graph.vs[vertex_id]["name"])
                    for vertex_id in vertex_id_path
                ]

                assert np.array_equal(
                    array_coords[0].as_array(),
                    self.path_factory.array_start_pos.as_array(),
                ), "Start point " "does not match"
                assert np.array_equal(
                    array_coords[-1].as_array(),
                    self.path_factory.array_goal_pos.as_array(),
                ), "Goal point does " "not match"

                local_coords = merged_graph.local_coords_from_array_coords(array_coords)

                # append start point and end point in local coords if not contained (
                # e.g. if min_flight_height%z_res != 0)
                if not np.array_equal(
                    local_coords[0].as_array(),
                    self.path_factory.local_start_pos.as_array(),
                ):
                    local_coords.insert(0, self.path_factory.local_start_pos)

                if not np.array_equal(
                    local_coords[-1].as_array(),
                    self.path_factory.local_goal_pos.as_array(),
                ):
                    local_coords.append(self.path_factory.local_goal_pos)

                path = Path(local_coords)
                # currently the coordinates are discrete according to x_res,
                # y_res and z_res, refine this
                path.interpolate_equal_spacing(self.path_factory.waypoint_resolution)

                dijkstra_paths.append(path)

            console.print(f"{int(comb_idx / number_iterations * 100)}% done.", end="\r")

        # check for doublet paths
        console.log(
            f"{len(dijkstra_paths)} were generated from {number_iterations} runs. If "
            f"the number of paths are significantly less then the number of "
            f"combinations, consider decreasing the number of combinations, otherwise "
            f"increase."
        )

        return dijkstra_paths, cost_values

    def get_merged_graph(self, grid_graphs: dict[str, GridGraph]) -> GridGraph:
        config = get_config()
        rows = int(self.scenario.y_length / self.scenario.y_res)
        columns = int(self.scenario.x_length / self.scenario.x_res)
        layers = int(self.scenario.z_length / self.scenario.z_res)
        merged_objective_str = "".join([f"{key}_" for key in grid_graphs])

        graph_name = (
            f"{merged_objective_str}graph_rows_{rows}_cols_{columns}_lays_{layers}"
        )

        path_to_grid_graph = get_grid_graph_path(
            DataManager().data_processing_path, self.scenario.scenario_info, graph_name
        )
        if os.path.exists(path_to_grid_graph):
            console.log(f"Found saved graph in {graph_name}. Load it.")
            start_load_time = time.time()
            merged_graph = load_grid_graph(
                DataManager().data_processing_path,
                self.scenario.scenario_info,
                graph_name,
            )
            console.log("Loading done.")
            console.log(f"Loading took {time.time() - start_load_time} seconds")
        else:
            start_calc_time = time.time()
            console.log("Merging and clipping graphs.")
            merged_graph = clip_and_merge_graphs(
                grid_graphs, config.preprocessor.normalization_type
            )
            console.log("Merging and clipping done.")
            console.log(
                f"Merging and clipping took {time.time() - start_calc_time} seconds"
            )
            console.log("Building done. Saving.")
            save_grid_graph(
                merged_graph,
                DataManager().data_processing_path,
                self.scenario.scenario_info,
                graph_name,
            )
        self.clipped_num_graph_edges = len(merged_graph.graph.es)
        self.clipped_num_graph_nodes = len(merged_graph.graph.vs)
        return merged_graph
