import os
import shutil
import unittest

import numpy as np
import yaml

from vias.config import load_config
from vias.grid_graph import save_grid_graph
from vias.grid_map import GridMap
from vias.path import Path
from vias.scenario import Scenario
from vias.simulators.simulator_grid_based import SimulatorGridBased
from vias.utils.helpers import ScenarioInfo


def get_test_folder_path():
    test_folder_path = os.path.join(os.getcwd(), "test_folder")
    return test_folder_path


def create_test_folder():
    test_folder_path = get_test_folder_path()
    if os.path.exists(test_folder_path):
        raise AssertionError("Delete old test folder before creating a new one")
    os.makedirs(test_folder_path)
    return test_folder_path


def delete_test_folder():
    test_folder_path = get_test_folder_path()
    if not os.path.exists(test_folder_path):
        raise AssertionError("Create test folder path before deleting it")
    shutil.rmtree(test_folder_path)


class TestSimulatorGridBased(unittest.TestCase):
    def setUp(self):
        self.test_folder_path = create_test_folder()
        self.create_test_config()

        self.create_test_scenario()

        self.create_test_map()

    def create_test_scenario(self):
        self.scenario_info = ScenarioInfo(
            map_NW_origin_lon=2.5,
            map_NW_origin_lat=40,
            x_length=20,
            y_length=20,
            z_length=10,
            x_res=4,
            y_res=4,
            z_res=5,
            user_identifier="test_scenario",
            min_flight_height=0,
            max_flight_height=30,
        )
        Scenario(self.scenario_info)

    def create_test_map(self):
        def f(x, y, z):
            return 1 + 2 * x

        grid_map = GridMap(self.scenario_info.convert_to_map_info("test_map"))
        grid_map.set_from_func(f)
        save_grid_graph(grid_map, self.test_folder_path, self.scenario_info, "test_map")

    def create_test_config(self):
        config_data = {"test_map": {"dimension": 3}}
        with open(os.path.join(self.test_folder_path, "test_config.yml"), "w") as file:
            yaml.dump(config_data, file, default_flow_style=False)
        load_config(os.path.join(self.test_folder_path, "test_config.yml"))

    def test_simulate(self):
        vec_x = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        vec_y = np.zeros(vec_x.shape)
        vec_z = np.zeros(vec_x.shape)
        path = Path([vec_x, vec_y, vec_z])
        simulator = SimulatorGridBased(identifier_str="test_sim", parameters={})
        simulator.load_inputs_and_init(["test_map"], self.test_folder_path)
        line_integral = simulator.simulate(path)
        self.assertEqual(line_integral, 156.0)

    def tearDown(self):
        delete_test_folder()
