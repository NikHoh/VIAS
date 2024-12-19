import math
from typing import Tuple, Dict, List

import numpy as np

from src.vias.path import Path
from src.vias.simulators.simulator import Simulator

from vias.simulators.graph_based import GraphBased

debug = False

# drone parameters
average_mass = 1.2  # kg
r_rotors = 0.1  # m
A_rotor = math.pi * r_rotors ** 2  # m²
number_rotors = 4
v_cruise = 14  # m/s
v_climb = 2  # m/s
v_descend = 1  # m/s

# physical parameters
g = 9.81  # m/(s²)
rho_air = 1.225  # kg/(m³)

# hover constant
c_hover = math.sqrt((average_mass ** 3 * g ** 3) / (2 * number_rotors * rho_air * A_rotor))
# drag coefficient
c_d = 0.6  # 0.0613
# drag area
A_drag = 0.1  # 0.029
# drag constant
c_drag = 0.5 * rho_air * c_d * A_drag

# climb constant
c_climb = 0.5 * average_mass * g * math.sqrt(
    (2 * average_mass * g / (rho_air * number_rotors * A_rotor)) + v_climb ** 2) + 3 / 2 * average_mass * g * v_climb

# constants for curve flight energy
C_theta = c_drag / (average_mass * g)
C_vs = 2 * average_mass * g / (rho_air * math.pi * number_rotors * r_rotors ** 2)


def get_delta_time(path: Path) -> np.ndarray:
    """Calculates time stamps for 3D path assuming a constant velocity v_cruise"""
    global v_cruise
    t = np.linalg.norm(np.diff(path.as_array()[:, 0:2], axis=0), axis=1) / v_cruise
    b_z = np.diff(path.vec_z, axis=0)
    t_z = np.where(b_z < 0, np.abs(b_z / v_descend), np.abs(b_z / v_climb))
    T = np.maximum(t, t_z)
    return T


def get_velocities_world_fixed(path: Path, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates roll and pitch velocities from the perspective of a world fixed coordinate frame"""
    V_x = np.diff(path.vec_x, axis=0) / T
    V_y = np.diff(path.vec_y, axis=0) / T
    return V_x, V_y


class SimulatorEnergy(Simulator, GraphBased):
    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)  # do not rely on super().__init__(**kwargs) here as kwargs get lost in MRO
        GraphBased.__init__(self, **kwargs, directed=True)  # do not rely on super().__init__(**kwargs) here as kwargs get lost in MRO

        self.cost_z_dir = self.z_dist * 10.0
        self.cost_z_dir_way_back = self.z_dist * 15.0
        self.cost_diag_in_xz_plane = math.sqrt(self.x_dist ** 2 + (self.z_dist * 10.0) ** 2)
        self.cost_diag_in_xz_plane_way_back = math.sqrt(self.x_dist ** 2 + (self.z_dist * 15.0) ** 2)
        self.cost_diag_in_yz_plane = math.sqrt(self.y_dist ** 2 + (self.z_dist * 10.0) ** 2)
        self.cost_diag_in_yz_plane_way_back = math.sqrt(self.y_dist ** 2 + (self.z_dist * 15.0) ** 2)
        self.cost_diag_in_xyz = math.sqrt(self.xy_dist ** 2 + (self.z_dist * 10.0) ** 2)
        self.cost_diag_in_xyz_way_back = math.sqrt(self.xy_dist ** 2 + (self.z_dist * 15.0) ** 2)

    def load_inputs(self, inputs_str: List[str], input_data_folder: str):
        pass

    def init(self):
        pass

    def simulate(self, path: Path) -> float:

        T = get_delta_time(path)  # TODO compare T with later t

        # for vertical movements
        dh = np.diff(path.vec_z, axis=0)
        E_climb = 0
        E_hover_descend = 0
        for time_id, dt in enumerate(T):
            if dh[time_id] > 0:
                v = dh[time_id] / dt
                E_climb += (0.5 * average_mass * g * math.sqrt(
                    (2 * average_mass * g / (
                            rho_air * number_rotors * A_rotor)) + v ** 2) + 3 / 2 * average_mass * g * v) * dt
            else:
                E_hover_descend += c_hover * dt

        # for horizontal movements
        #   a) energy for acceleration
        E_accel = 0.5 * average_mass * v_cruise ** 2

        #   b) energy for pitch and roll actions
        V_x, V_y = get_velocities_world_fixed(path, T)
        E_curve = 0

        for idx, (v_x, v_y) in enumerate(zip(V_x, V_y)):
            if v_x == 0:
                theta = 0
            else:
                theta = -math.copysign(1, v_x) * math.atan(
                    C_theta * (v_x ** 2 + v_y ** 2) / math.sqrt(1 + v_y ** 2 / (v_x ** 2)))  # formula red box 1
            if theta == 0:
                phi = math.copysign(1, v_y) * math.atan(c_drag * v_y ** 2 / (average_mass * g))
            else:
                phi = math.atan(-v_y / v_x * math.sin(theta))  # formula red box 2
            if theta != 0:
                assert math.copysign(1, v_x) == -math.copysign(1,
                                                               theta), f"v_x is {v_x} but theta is {theta}"
            assert math.copysign(1, v_y) == math.copysign(1,
                                                          phi), f"v_y is {v_y} but phi is {phi}"
            assert theta >= -math.pi / 4 and theta < math.pi / 4, f" pitch angle theta {theta} exceeds bounds"
            assert phi >= -math.pi / 4 and phi < math.pi / 4, f" roll angle theta {phi} exceeds bounds"
            v_s = math.sqrt(C_vs / (math.cos(theta) * math.cos(phi)) + (
                    math.cos(phi) * math.sin(theta) * v_x - math.sin(phi) * v_y) ** 2)
            E_curve += average_mass * g * (v_s + math.cos(phi) * math.sin(theta) * v_x - math.sin(phi) * v_y) / (
                    2 * math.cos(theta) * math.cos(phi)) * T[idx]

        simulator_result = E_climb + E_hover_descend + E_accel + E_curve

        return simulator_result

    def add_edge(self, from_node: Tuple, to_node: Tuple, node_dict: Dict, edges: List, edge_weights: List, x_dir=False,
                 y_dir=False, z_dir=False, diag_in_xy_plane=False, diag_in_xz_plane=False, diag_in_yz_plane=False,
                 diag_in_xyz=False, way_back=False):
        p0 = node_dict[from_node]
        p1 = node_dict[to_node]

        if x_dir:
            cell_cost = self.x_dist
        elif y_dir:
            cell_cost = self.y_dist
        elif z_dir:
            if not way_back:  # climbing
                cell_cost = self.cost_z_dir
            else:  # descending
                cell_cost = self.cost_z_dir_way_back
        elif diag_in_xy_plane:
            cell_cost = self.xy_dist
        elif diag_in_xz_plane:
            if not way_back:  # climbing
                cell_cost = self.cost_diag_in_xz_plane
            else:  # descending
                cell_cost = self.cost_diag_in_xz_plane_way_back
        elif diag_in_yz_plane:
            if not way_back:  # climbing
                cell_cost = self.cost_diag_in_yz_plane
            else:  # descending
                cell_cost = self.cost_diag_in_yz_plane_way_back
        elif diag_in_xyz:
            if not way_back:  # climbing
                cell_cost = self.cost_diag_in_xyz
            else:  # descending
                cell_cost = self.cost_diag_in_xyz_way_back
        else:
            assert False, "Not defined"


        assert cell_cost > 0, f"Cost is {cell_cost} <= 0, which is not allowed."
        edges.append([p0, p1])
        edge_weights.append(cell_cost)
