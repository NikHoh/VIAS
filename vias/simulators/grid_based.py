from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional

import numpy as np
import igraph
from vias.grid_graph import GridGraph
from vias.grid_map import GridMap
from vias.simulators.graph_based import GraphBased
from vias.utils.helpers import ArrayCoord


class GridBased(GraphBased):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, directed=False)
        self._grid_map: Optional[GridMap] = None


    @property
    def grid_map(self):
        if self._grid_map is None:
            self._grid_map = self.load_grid_map()
        return self._grid_map

    @abstractmethod
    def load_grid_map(self) -> GridMap:
        raise NotImplementedError("Users must implement load_grid_map() to use this base class.")


    def _generate_grid_graph(self, rows, columns, layers, graph_name, nfa_graph: Optional[igraph.Graph] = None) -> GridGraph:
        grid_graph = super()._generate_grid_graph(rows, columns, layers, graph_name, nfa_graph)
        self.scenario.grid_maps[self.grid_map.name].graph = grid_graph
        return grid_graph

    def add_edge(self, from_node: Tuple, to_node: Tuple, node_dict: Dict, edges: List, edge_weights: List, x_dir=False,
                 y_dir=False, z_dir=False, diag_in_xy_plane=False, diag_in_xz_plane=False, diag_in_yz_plane=False,
                 diag_in_xyz=False, way_back=False) -> None:
        grid_map = self.grid_map

        p0 = node_dict[from_node]
        p1 = node_dict[to_node]

        if x_dir:
            dist = self.x_dist
        elif y_dir:
            dist = self.y_dist
        elif z_dir:
            dist = self.z_dist
        elif diag_in_xy_plane:
            dist = self.xy_dist
        elif diag_in_xz_plane:
            dist = self.xz_dist
        elif diag_in_yz_plane:
            dist = self.yz_dist
        elif diag_in_xyz:
            dist = self.xyz_dist
        else:
            assert False, "Not defined"

        # trapezoidal rule
        grid_values = grid_map.get_values_from_array_coords([ArrayCoord(*from_node),
                                                                        ArrayCoord(*to_node)])
        if np.any(grid_values == 0):
            return
        cell_cost = 0.5 * np.sum(grid_values) * dist

        assert cell_cost > 0, f"Cost is {cell_cost} <= 0, which is not allowed."
        edges.append([p0, p1])
        edge_weights.append(cell_cost)