from typing import Optional, List, Union

import geomdl
import matplotlib.pyplot as plt
import numpy as np
from geomdl.fitting import approximate_curve
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.interpolate import interp1d
from geomdl.NURBS import Curve as NurbsCurve

from vias.osm_exporter import global_coord_from_tmerc_coord
from vias.utils.helpers import LocalCoord
from vias.utils.tools import euclidian_distance
import copy as cp


class ChordalNURBS(NurbsCurve):
    """Class that returns curve.evalpts that are equally distributed in the operation space (i.e., chordal parametrization)"""

    def __init__(self, waypoint_resolution):
        super().__init__()
        self.harmonic_knots: Optional[np.array] = None
        self.waypoint_resolution = waypoint_resolution

    @property
    def evalpts(self):
        """ Evaluated points.

        Please refer to the `wiki <https://github.com/orbingol/NURBS-Python/wiki/Using-Python-Properties>`_ for details
        on using this class member.

        :getter: Gets the coordinates of the evaluated points
        :type: list
        """
        if self._eval_points is None or len(self._eval_points) == 0:
            u = np.arange(0, 1001) / 1000
            eval_p = self.evaluate_list(list(u))
            y = np.cumsum(np.linalg.norm(np.diff(eval_p, axis=0), axis=1))  # summed distances at y axis
            y = np.append(0, y)

            num_points = np.ceil(y[-1] / self.waypoint_resolution).astype(int)
            y_new = np.linspace(y[0], y[-1], num_points)

            interp = interpolate.interp1d(y, u)
            u_new = interp(y_new)
            self.harmonic_knots = u_new

            self._eval_points = self.evaluate_list(list(u_new))
        else:
            self._eval_points = self.evaluate_list(list(self.harmonic_knots))
        return self._eval_points


class Path(object):
    def __init__(self, input_object: Union[ChordalNURBS, List[np.ndarray], List[LocalCoord]]):
        """Either gets NURBS_harmonic or list of LocalCoords (waypoints) or gets list of vec_x, vec_y, vec_z numpy arrays"""
        self._vec_x: Optional[np.array] = None
        self._vec_y: Optional[np.array] = None
        self._vec_z: Optional[np.array] = None
        self._waypoint_list: Optional[List[LocalCoord]] = None
        self._nurbs_curve: Optional[ChordalNURBS] = None
        self._length: Optional[float] = None

        if isinstance(input_object, ChordalNURBS):
            self._nurbs_curve = input_object
        elif isinstance(input_object, list):
            assert len(input_object) > 0, "Empty input not allowed"
            if isinstance(input_object[0], np.ndarray):
                assert len(input_object) == 3, "Expected three numpy arrays as input (vec_x, vec_y, vec_z)."
                self._vec_x = input_object[0]
                assert isinstance(input_object[1], np.ndarray), "Expected vec_y to be numpy array"
                assert input_object[1].shape == self.vec_x.shape, "Expect vec_y to have same shape and length as vec_X"
                self._vec_y = input_object[1]
                assert isinstance(input_object[2], np.ndarray), "Expected vec_z to be numpy array"
                assert input_object[2].shape == self.vec_y.shape, "Expect vec_y to have same shape and length as vec_X"
                self._vec_z = input_object[2]
            elif isinstance(input_object[0], LocalCoord):
                self._waypoint_list = input_object

    @property
    def vec_x(self):
        if self._vec_x is None:
            self._vec_x = [local_coord.x for local_coord in
                           self.waypoint_list]  # Caution: calling waypoint_list property
        return self._vec_x

    @vec_x.setter
    def vec_x(self, vec_x: np.ndarray):
        self._vec_x = None
        self._waypoint_list = None
        self._nurbs_curve = None
        self._vec_x = vec_x

    @property
    def vec_y(self):
        if self._vec_y is None:
            self._vec_y = [local_coord.y for local_coord in
                           self.waypoint_list]  # Caution: calling waypoint_list property
        return self._vec_y

    @vec_y.setter
    def vec_y(self, vec_y: np.ndarray):
        self._vec_y = None
        self._waypoint_list = None
        self._nurbs_curve = None
        self._vec_y = vec_y

    @property
    def vec_z(self):
        if self._vec_z is None:
            self._vec_z = [local_coord.z for local_coord in
                           self.waypoint_list]  # Caution: calling waypoint_list property
        return self._vec_z

    @vec_z.setter
    def vec_z(self, vec_z: np.ndarray):
        self._vec_z = None
        self._waypoint_list = None
        self._nurbs_curve = None
        self._vec_z = vec_z

    @property
    def waypoint_list(self) -> List[LocalCoord]:
        if self._waypoint_list is None:
            if self._vec_x is not None and self._vec_y is not None and self._vec_z is not None:
                self._waypoint_list = [LocalCoord(*arr) for arr in np.vstack((self._vec_x, self._vec_y, self._vec_z)).T]
            elif self._nurbs_curve is not None:
                self._waypoint_list = [LocalCoord(*pts) for pts in self._nurbs_curve.evalpts]
            else:
                assert False, "Set waypoint_list, or vec_x, vec_y, and vec_z, or nurbs_curve before retrieving path element."
        return self._waypoint_list

    @property
    def bee_line_distance(self):
        return euclidian_distance(self.waypoint_list[0].as_array(), self.waypoint_list[-1].as_array())

    @waypoint_list.setter
    def waypoint_list(self, waypoint_list: List[LocalCoord]):
        self._vec_x = None
        self._vec_y = None
        self._vec_z = None
        self._nurbs_curve = None
        self._waypoint_list = waypoint_list

    @property
    def nurbs_curve(self):
        if self._nurbs_curve is None:
            assert False, "Call self.approximate_nurbs() first to be able to get nurbs_curve"
        return self._nurbs_curve

    @nurbs_curve.setter
    def nurbs_curve(self, nurbs_curve):
        self._vec_x = None
        self._vec_y = None
        self._vec_z = None
        self._waypoint_list = None
        self._nurbs_curve = nurbs_curve

    @property
    def length(self):
        self._length = float(np.sum(np.linalg.norm(np.diff(self.as_array(), axis=0), axis=1)))
        return self._length

    def interpolate_equal_spacing(self, delta=1.0):
        distances = np.linalg.norm(np.diff(self.as_array(), axis=0),
                                   axis=1)  # Euclidean distances between consecutive points
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Cumulative distance along the path

        num_points = np.ceil(self.length / delta).astype(int)
        equal_distances = np.linspace(0, cumulative_distances[-1], num_points)

        for obj_str in ["vec_x", "vec_y", "vec_z"]:
            interp_func = interp1d(cumulative_distances, self.__getattribute__(obj_str), kind='linear')
            self.__setattr__(obj_str, interp_func(equal_distances))

    def smooth(self, num_rounds):
        kernel = np.array([1, 2, 1])
        kernel = kernel / kernel.sum()  # Normalize the kernel

        for _ in range(num_rounds):  # three smoothing rounds
            points = np.copy(self.as_array())
            smoothed_points = np.copy(self.as_array())

            # Apply the binomial kernel to x, y, z coordinates independently
            for i in range(3):  # For each dimension (x, y, z)
                smoothed_inner = np.convolve(points[:, i], kernel, mode='same')  # Exclude first and last
                smoothed_points[2:-2, i] = smoothed_inner[
                                           2:-2]  # Keep the first and last two points unchanged to avoid convolution boundary effects

            self.vec_x = smoothed_points[:, 0]
            self.vec_y = smoothed_points[:, 1]
            self.vec_z = smoothed_points[:, 2]

    def approximate_nurbs(self, num_cp_for_approx: int, nurbs_order: int, waypoint_resolution: int):
        # Make sure that path consists of more waypoints then wanted number of control points for its approximation
        if num_cp_for_approx >= len(self.waypoint_list):
            desired_num_waypoints = int(2 * num_cp_for_approx)
            desired_delta = self.length / desired_num_waypoints
            self.interpolate_equal_spacing(
                delta=desired_delta)  # the finer grained path will be set back to its normal graining after the approximation

        assert len(self.waypoint_list) > num_cp_for_approx, f"Something went wrong in the interpolation. Length waypoint list ist {len(self.waypoint_list)}, the number of control points for approximation is {num_cp_for_approx}. The length of the path is {self.length}."
        approximated_nurbs_curve = approximate_curve([local_coord.as_array() for local_coord in self.waypoint_list],
                                                     nurbs_order - 1,
                                                     ctrlpts_size=num_cp_for_approx)
        self.vec_x = None
        self.vec_y = None
        self.vec_z = None
        self.waypoint_list = None

        chordal_nurbs_curve = ChordalNURBS(waypoint_resolution)
        chordal_nurbs_curve.degree = approximated_nurbs_curve.degree
        chordal_nurbs_curve.ctrlpts = approximated_nurbs_curve.ctrlpts
        chordal_nurbs_curve.weights = approximated_nurbs_curve.weights
        chordal_nurbs_curve.knotvector = approximated_nurbs_curve.knotvector

        self.nurbs_curve = chordal_nurbs_curve

    # def refine(self, point_distance_threshold):
    #     diffs = np.cumsum(np.linalg.norm(np.diff(self.waypoint_list, axis=0), axis=1))  # summed distances at y axis
    #     diffs = np.append(0, diffs)
    #
    #     new_diffs = np.arange(0, self.length, point_distance_threshold)
    #     new_diffs = np.append(new_diffs, self.length)
    #     new_diffs = new_diffs.tolist()
    #     new_diffs.extend(diffs.tolist())
    #     new_diffs = list(set(new_diffs))
    #     new_diffs.sort()
    #     new_diffs = np.array(new_diffs)
    #
    #     interp_x = interpolate.interp1d(diffs, self.vec_x, fill_value=np.nan, bounds_error=False)
    #     interp_y = interpolate.interp1d(diffs, self.vec_y, fill_value=np.nan, bounds_error=False)
    #     interp_z = interpolate.interp1d(diffs, self.vec_z, fill_value=np.nan, bounds_error=False)
    #     vec_x = interp_x(new_diffs)
    #     vec_x = vec_x[~np.isnan(vec_x)]
    #
    #     vec_y = interp_y(new_diffs)
    #     vec_y = vec_y[~np.isnan(vec_y)]
    #
    #     vec_z = interp_z(new_diffs)
    #     vec_z = vec_z[~np.isnan(vec_z)]
    #
    #     waypoint_list = list(zip(vec_x, vec_y, vec_z))
    #     waypoint_list = list(dict.fromkeys(waypoint_list))  # remove duplicates
    #
    #     self.vec_x = np.array([el[0] for el in waypoint_list])
    #     self.vec_y = np.array([el[1] for el in waypoint_list])
    #     self.vec_z = np.array([el[2] for el in waypoint_list])
    #
    #     self.point_weights = np.full(self.vec_x.shape, 1.0)
    #     self.calculate_waypoint_vector()

    def as_array(self):
        return np.vstack((self.vec_x, self.vec_y, self.vec_z)).T

    def plot(self, hold=False, fig=None, ax=None, three_dim=True):
        if self.nurbs_curve is not None:
            ctrlpts = np.array(self.nurbs_curve.ctrlpts)
        else:
            ctrlpts = None
        curvepts = np.array(self.waypoint_list)

        # Draw the control points polygon, the 3D curve and the vectors
        if fig is None:
            fig = plt.figure(figsize=(10.67, 8), dpi=96)
        if ax is None:
            ax = Axes3D(fig)
            ax.set_aspect('auto')

        # Plot 3D lines
        if ctrlpts is not None:
            if three_dim:
                ax.set_box_aspect((np.ptp(ctrlpts[:, 0]), np.ptp(ctrlpts[:, 1]), np.ptp(ctrlpts[:, 2])))
                ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], ctrlpts[:, 2], color='black', linestyle='-.', marker='o',
                        linewidth=1)
            else:
                ax.set_box_aspect((np.ptp(ctrlpts[:, 0]), np.ptp(ctrlpts[:, 1])))
                ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], color='black', linestyle='-.', marker='o',
                        linewidth=1)
        else:
            if three_dim:
                ax.set_box_aspect((np.ptp(curvepts[:, 0]), np.ptp(curvepts[:, 1]), np.ptp(curvepts[:, 2])))
        if three_dim:
            ax.plot(curvepts[:, 0], curvepts[:, 1], curvepts[:, 2], color='brown', linestyle='-', linewidth=2)
        else:
            ax.plot(curvepts[:, 0], curvepts[:, 1], color='brown', linestyle='-', linewidth=2)
        if not hold:
            plt.show()
            plt.close()

    def path_to_npy(self, size=None):
        if not size is None:
            orig_arr = self.as_array()
            arr = np.full((size, 3), np.nan)
            arr[0:orig_arr.shape[0], :] = orig_arr
            return arr

        return self.as_array()

    def to_lon_lat_height(self, grid_graph: 'GridGraph') -> List[np.ndarray]:
        lon_vec = []
        lat_vec = []
        height_vec = []
        for local_coord in self.waypoint_list:
            tmerc_coord = grid_graph.tmerc_coord_from_local_coord(local_coord)
            global_coord = global_coord_from_tmerc_coord(tmerc_coord)
            lon_vec.append(global_coord.lon)
            lat_vec.append(global_coord.lat)
            height_vec.append(local_coord.z)
        return [np.array(lon_vec), np.array(lat_vec), np.array(height_vec)]

    def insert_takoff_landing_sequence(self, takeoff_sequence: List[LocalCoord], landing_sequence: List[LocalCoord]):
        current_waypoints = cp.deepcopy(self.waypoint_list)
        new_waypoint_list = takeoff_sequence + current_waypoints + landing_sequence
        self.waypoint_list = new_waypoint_list






