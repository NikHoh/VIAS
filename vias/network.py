import itertools
import math
from collections import OrderedDict

import networkx as nx

# import config
from vias.utils.tools import euclidian_distance


def fuse_paths(paths):
    fused_paths = paths.pop(0)
    for path in paths:
        for idx, component in enumerate(path):
            fused_paths[idx] = list(fused_paths[idx])
            fused_paths[idx].extend(component)
    return [fused_paths]

class Network(nx.DiGraph):
    """This class inherits from the Graph class from networkx and extends it with mainly a distance matrix."""

    def __init__(self):
        super().__init__()
        self.travel_time: float = 0
        self.link_usage: list[int] = []
        self.distance_matrix = None
        self.nearest_neighbor_matrix = None

    def __len__(self):
        return len(self.nodes)

    def calculate_distance_matrix(self, style="euclid"):
        nodes = list(nx.get_node_attributes(self, 'pos').values())
        distance_matrix = []
        for i in range(0, len(nodes)):
            b = []
            for j in range(0, len(nodes)):
                b.append(None)
            distance_matrix.append(b)

        for node1, node2 in itertools.product(nodes, nodes):
            if style == "euclid":
                dist = math.ceil(math.sqrt(
                    (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2))
                assert dist is not None
                distance_matrix[nodes.index(node1)][nodes.index(node2)] = dist
            elif style == "manhattan":
                dist = int(
                    abs(node1[0] - node2[0]) + abs(node1[1] - node2[1]))
                distance_matrix[nodes.index(node1)][nodes.index(node2)] = dist
                assert dist is not None
        self.distance_matrix = distance_matrix

    def calculate_nearest_neighbor_matrix(self):
        nn_matrix = []
        for id1, _ in enumerate(self.vport_positions):
            nn_matrix.append(OrderedDict())
            for id2, _ in enumerate(self.vport_positions):
                if id2 == id1:
                    continue
                nn_matrix[id1][id2] = self.distance_matrix[id1][id2]
        # sort the OrderedDicts
        for ind, _ in enumerate(self.vport_positions):
            nn_matrix[ind] = OrderedDict(
                sorted(nn_matrix[ind].items(), key=lambda t: t[1]))
        return nn_matrix

    def get_paths(self, separated=False, grid_res=None, only_cp=False):
        """Returns a list consisting of the vectors (vec_x, vec_y, vec_z) containing the path coordinates respectively."""
        # raise NotImplementedError('Users must define optimize to use this base class.')
        paths = []
        path = []
        for idx, edge in enumerate(self.edges):
            if idx == 0:
                path = [self.nodes[edge[0]]['pos'], self.nodes[edge[1]]['pos']]
            else:
                # only append last point
                path.append(self.nodes[edge[1]]['pos'])

        path = list(zip(*path))
        paths.append(path)
        # Attention: this will not work for more than one path in the network
        if not separated:
            return fuse_paths(paths)
        return paths


    def add_point_chain(self, start_id, start_pos, points, end_id, end_pos):
        """Important, given start_pos and end_pos should not be part of the points vector, as they are assummed to
        have been already added."""
        node_id = len(self.nodes)
        link_id = len(self.edges)
        # node_id = start_id  # in case that points is empty
        assert points[0] != tuple(start_pos), "Given start_pos and end_pos should not be part of the points vector."
        point = start_pos  # in case that points is empty
        for idx, point in enumerate(points):
            self.add_node(node_id, pos=point)
            if idx > 0:
                link_length = euclidian_distance(point, points[idx - 1])
                self.add_edge(node_id - 1, node_id, id=link_id, length=link_length, capacity=28800)
            else:
                link_length = euclidian_distance(point, start_pos)
                self.add_edge(start_id, node_id, id=link_id, length=link_length, capacity=28800)
            node_id += 1
            link_id += 1
        # add last link
        assert tuple(end_pos) != point, "Given start_pos and end_pos should not be part of the points vector."
        link_length = euclidian_distance(point, end_pos)
        self.add_edge(node_id - 1, end_id, id=link_id, length=link_length, capacity=28800)
