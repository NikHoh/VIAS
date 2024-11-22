import math

import numpy as np

from vias.path import Path
from vias.path_factory import PathFactory
from vias.scenario import Scenario
from vias.utils.helpers import LocalCoord


class Coder(object):
    """Class that manages the transformation from a set of vertiport positions to a Nurbswork representation (encode)
    the transformation from an individual's gene code representation back to a Nurbswork (decode).
    If there is a link (NURBS curve) between a set of points is determined by the parameter "link_mask" and can not
    be changed after a CoderNurbswork object has been initialized."""

    def __init__(self):
        self.path_factory = PathFactory()

        self.adaptive_weights = False
        self.use_z_component = True

        if self.adaptive_weights and self.use_z_component:
            self.size_gene_per_cp = 4
        elif self.adaptive_weights or self.use_z_component:
            self.size_gene_per_cp = 3
        else:
            self.size_gene_per_cp = 2

    @property
    def size_individual(self):
        return self.size_gene_per_cp * self.path_factory.num_variable_control_points

    def encode(self, path: Path) -> np.ndarray:
        """
        Calculates a nurbswork blueprint (nurbswork without links, just nodes) from the vertiport positions. Links_mask
        is a binary string, that indicates if there is a link between a pair of vport_positions. It is usually
        calculated and saved by the GA Optimizer and can be set in the config.yaml.
        """

        control_points = path.nurbs_curve.ctrlpts[1:-1]  # leave out fixed control points
        weights = path.nurbs_curve.weights[1:-1]  # leave out fixed control points
        chunked_control_points = np.array(control_points)
        if self.use_z_component and self.adaptive_weights:
            chunked_individual = np.hstack((chunked_control_points, weights))
            individual = chunked_individual.reshape(1, -1)
        elif self.use_z_component:
            individual = chunked_control_points.reshape(1, -1)
        elif self.adaptive_weights:
            chunked_control_points = chunked_control_points[:, 0:2]  # leave out z-column
            chunked_individual = np.hstack((chunked_control_points, weights))
            individual = chunked_individual.reshape(1, -1)
        else:
            chunked_control_points = chunked_control_points[:, 0:2]
            individual = chunked_control_points.reshape(1, -1)
        return individual

    def decode(self, individual: np.ndarray) -> Path:
        """
        Calculates a path from the individual's representation.
        If the weights are adaptive (adaptive_weights = True) and the z-component is used (use_z_component = True), then it
        looks like the individual looks like
        [x_1, y_1, z_1, w_1, x_2, y_2, z_2, w_2, ...].
        If the inner weights aren't considered (adaptive_weights = False) the individual looks like
        [x_1, y_1, z_1, x_2, y_2, z_2, ...].
        if additionally the z-component is not considered (use_z_component = False) the individual looks like
        [x_1, y_1, x_2, y_2, ...].
        """

        assert len(individual) == self.size_individual, "Expected individual size does not match."

        chunked_individual = np.array(individual).reshape(-1, self.size_gene_per_cp)
        cp_x_vec = chunked_individual[:, 0]
        cp_y_vec = chunked_individual[:, 1]
        if self.use_z_component and self.adaptive_weights:
            cp_z_vec = chunked_individual[:, 2]
            weight_vec = chunked_individual[:, 3]
        elif self.use_z_component:
            cp_z_vec = chunked_individual[:, 2]
            weight_vec = np.ones(cp_x_vec.shape)
        elif self.adaptive_weights:
            weight_vec = chunked_individual[:, 2]
            cp_z_vec = np.zeros(cp_x_vec.shape)
        else:
            cp_z_vec = np.zeros(cp_x_vec.shape)
            weight_vec = np.ones(cp_x_vec.shape)

        control_points = [LocalCoord(*col_vec) for col_vec in np.vstack((cp_x_vec, cp_y_vec, cp_z_vec)).T]

        path = self.path_factory.generate_path_from_variable_control_points(control_points, weight_vec)

        return path


    def get_bounds(self):
        """Calculates the bounds for the problem (position constraints for control points)"""
        scenario = Scenario()
        if self.use_z_component and self.adaptive_weights:
            low_gene_piece = [0.0, 0.0, 0.0, 0.0]
            up_gene_piece = [float(scenario.x_length), float(scenario.y_length), float(scenario.z_length), math.inf]
        elif self.use_z_component:
            low_gene_piece = [0.0, 0.0, 0.0]
            up_gene_piece = [float(scenario.x_length), float(scenario.y_length), float(scenario.z_length)]
        elif self.adaptive_weights:
            low_gene_piece = [0.0, 0.0, 0.0]
            up_gene_piece = [float(scenario.x_length), float(scenario.y_length), math.inf]
        else:
            low_gene_piece = [0.0, 0.0]
            up_gene_piece = [float(scenario.x_length), float(scenario.y_length)]
        low_bound = low_gene_piece * self.path_factory.num_variable_control_points
        up_bound = up_gene_piece * self.path_factory.num_variable_control_points
        assert len(low_bound) == self.size_individual
        return low_bound, up_bound
