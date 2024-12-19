import itertools
import math
import os
import time
from abc import abstractmethod

import igraph
from igraph import Graph as IGraph

from vias.config import get_config
from vias.console_manager import console
from vias.data_manager import DataManager
from vias.grid_graph import (
    GridGraph,
    get_grid_graph_path,
    load_grid_graph,
    save_grid_graph,
)
from vias.scenario import Scenario
from vias.simulators.function_category import FunctionCategory


class GraphBased(FunctionCategory):
    def __init__(self, **kwargs):
        self.identifier_str = kwargs["identifier_str"]
        self.directed = kwargs["directed"]
        self.scenario = Scenario()
        self._grid_graph: GridGraph | None = None
        self.x_dist = self.scenario.scenario_info.x_res
        self.y_dist = self.scenario.scenario_info.y_res
        self.z_dist = self.scenario.scenario_info.z_res
        self.xy_dist = math.sqrt(self.x_dist**2 + self.y_dist**2)
        self.xz_dist = math.sqrt(self.x_dist**2 + self.z_dist**2)
        self.yz_dist = math.sqrt(self.y_dist**2 + self.z_dist**2)
        self.xyz_dist = math.sqrt(self.x_dist**2 + self.y_dist**2 + self.z_dist**2)

    def derive_grid_graph(
        self, nfa_graph: igraph.Graph | None = None
    ) -> GridGraph | None:
        if self._grid_graph is None:
            g = self._get_grid_graph(nfa_graph)
            self._grid_graph = g
        return self._grid_graph

    @abstractmethod
    def add_edge(
        self,
        from_node: tuple[int, int, int],
        to_node: tuple[int, int, int],
        node_dict: dict,
        edges: list[list[int]],
        edge_weights: list[float],
        x_dir=False,
        y_dir=False,
        z_dir=False,
        diag_in_xy_plane=False,
        diag_in_xz_plane=False,
        diag_in_yz_plane=False,
        diag_in_xyz=False,
        way_back=False,
    ):
        raise NotImplementedError(
            "Users must define add_edge() to use this base class."
        )

    def _get_grid_graph(self, nfa_graph: igraph.Graph | None = None) -> GridGraph:
        get_config()
        rows = int(self.scenario.scenario_info.y_length / self.y_dist)
        columns = int(self.scenario.scenario_info.x_length / self.x_dist)
        layers = int(self.scenario.scenario_info.z_length / self.z_dist)
        graph_name = (
            f"{self.identifier_str}_graph_rows_{rows}_cols_{columns}_lays_{layers}"
        )

        path_to_grid_graph = get_grid_graph_path(
            DataManager().data_processing_path, self.scenario.scenario_info, graph_name
        )
        if os.path.exists(path_to_grid_graph):
            console.log(f"Found saved graph in {graph_name}. Load it.")
            start_load_time = time.time()
            g = load_grid_graph(
                DataManager().data_processing_path,
                self.scenario.scenario_info,
                graph_name,
            )
            console.log("Loading done.")
            console.log(f"Loading took {time.time() - start_load_time} seconds")
        else:
            start_calc_time = time.time()
            # get graph from grid
            console.log(f"Building graph {graph_name}. This may take a while...")
            g = self._generate_grid_graph(rows, columns, layers, graph_name, nfa_graph)
            console.log("Building done.")
            console.log(f"Building took {time.time() - start_calc_time} seconds")
            console.log("Building done. Saving.")
            save_grid_graph(
                g,
                DataManager().data_processing_path,
                self.scenario.scenario_info,
                graph_name,
            )
        return g

    def _generate_grid_graph(
        self, rows, columns, layers, graph_name, nfa_graph: igraph.Graph | None = None
    ) -> GridGraph:
        potential_array_coords = list(
            itertools.product(range(rows), range(columns), range(layers))
        )
        if nfa_graph is None:
            needed_array_coords = potential_array_coords
            nodes_to_delete = set()
        else:
            nodes_to_delete = set([array_coord for array_coord in nfa_graph.vs["name"]])
            needed_array_coords = [
                array_coord
                for array_coord in potential_array_coords
                if array_coord not in nodes_to_delete
            ]

        node_dict = dict(
            zip(needed_array_coords, range(0, len(needed_array_coords)), strict=False)
        )

        if layers == 1:
            dim = 2
        else:
            dim = 3

        g = IGraph(directed=self.directed)
        g.add_vertices(len(node_dict), attributes={"name": list(node_dict.keys())})
        edges: list[list[int]] = []
        edge_weights: list[float] = []

        last_row, last_column, last_layer = rows - 1, columns - 1, layers - 1

        for layer in range(layers):
            console.print(f"Building layer {layer + 1} of {layers}", end="\r")
            for row, column in itertools.product(range(rows), range(columns)):
                SW = S = SE = E = True
                UW = UN = UE = US = True
                UNE = USE = USW = UNW = True
                U = True
                if row == last_row:  # last row
                    SW = S = SE = False
                    USW = US = USE = False
                elif row == 0:  # first row
                    UNW = UN = UNE = False
                if column == 0:
                    SW = USW = False
                    UW = UNW = False
                elif column == last_column:
                    UE = UNE = False
                    E = False
                    SE = USE = False
                if layer == last_layer:
                    U = False
                    UN = UE = US = UW = False
                    UNW = UNE = USE = USW = False

                from_node = (row, column, layer)
                if from_node in nodes_to_delete:
                    continue
                if SW:  # /
                    to_node = (row + 1, column - 1, layer)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xy_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xy_plane=True,
                                way_back=True,
                            )
                if S:  # |
                    to_node = (row + 1, column, layer)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            y_dir=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                y_dir=True,
                                way_back=True,
                            )
                if SE:  # \
                    to_node = (row + 1, column + 1, layer)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xy_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xy_plane=True,
                                way_back=True,
                            )
                if E:  # -
                    to_node = (row, column + 1, layer)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            x_dir=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                x_dir=True,
                                way_back=True,
                            )
                if U:  # add vertical links
                    # vertical one (U)
                    to_node = (row, column, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            z_dir=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                z_dir=True,
                                way_back=True,
                            )
                if UN:
                    to_node = (row - 1, column, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_yz_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_yz_plane=True,
                                way_back=True,
                            )
                if UE:
                    to_node = (row, column + 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xz_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xz_plane=True,
                                way_back=True,
                            )
                if US:
                    to_node = (row + 1, column, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_yz_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_yz_plane=True,
                                way_back=True,
                            )
                if UW:
                    to_node = (row, column - 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xz_plane=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xz_plane=True,
                                way_back=True,
                            )
                if USE:
                    to_node = (row + 1, column + 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xyz=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xyz=True,
                                way_back=True,
                            )
                if UNE:
                    to_node = (row - 1, column + 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xyz=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xyz=True,
                                way_back=True,
                            )
                if UNW:
                    to_node = (row - 1, column - 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xyz=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xyz=True,
                                way_back=True,
                            )
                if USW:
                    to_node = (row + 1, column - 1, layer + 1)
                    if to_node not in nodes_to_delete:
                        self.add_edge(
                            from_node,
                            to_node,
                            node_dict,
                            edges,
                            edge_weights,
                            diag_in_xyz=True,
                        )
                        if self.directed:
                            self.add_edge(
                                to_node,
                                from_node,
                                node_dict,
                                edges,
                                edge_weights,
                                diag_in_xyz=True,
                                way_back=True,
                            )

        g.add_edges(edges)
        normalized_edge_weights = edge_weights

        g.es[self.identifier_str] = normalized_edge_weights

        # delete isolated nodes from graph
        vertices = g.vs
        degrees = igraph.Graph.degree(g)
        isolated_nodes = [v.index for v in vertices if degrees[v.index] == 0]
        g.delete_vertices(isolated_nodes)

        grid_graph = GridGraph(
            self.scenario.scenario_info.convert_to_map_info(graph_name),
            g,
            dimension=dim,
        )

        return grid_graph
