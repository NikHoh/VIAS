# mopp == Multi-Objective Path Planning
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime as dt

import numpy as np

from vias.config import load_config
from vias.console_manager import console, start_live_context
from vias.constraint_checkers.constraint_checker import ConstraintChecker
from vias.data_manager import DataManager
from vias.evaluator import Evaluator
from vias.path_factory import PathFactory
from vias.preprocessor import PreProcessor
from vias.scenario import Scenario
from vias.simulators.graph_based import GraphBased
from vias.simulators.non_graph_based import NonGraphBased
from vias.simulators.simulator import Simulator
from vias.utils.helpers import (
    GlobalCoord,
    NoSuitableApproximation,
    OptimizationEndsInNiches,
    ScenarioInfo,
    get_mopp_identifier,
    load_scenario_info_from_json,
)


def main(
    base_data_folder: str,
    path_start_lon: float,
    path_start_lat: float,
    path_goal_lon: float,
    path_goal_lat: float,
    path_to_config: str,
    data_input_folder: str,
    data_output_folder: str,
    data_processing_folder: str,
    **kwargs,
):
    """

    :param map_NW_origin_lon:
    :param map_NW_origin_lat:
    :param x_length:
    :param y_length:
    :param user_identifier: A string that is part of the output folder name generated
    in **data_output_folder**
    :param path_start_lon:
    :param path_start_lat:
    :param path_end_lon:
    :param path_end_lat:
    :param path_to_config:
    :param data_input_folder:
    :param data_output_folder: Path to the general output_folder where the specific
    output folder for this run is created.
    :return:
    """
    live = start_live_context()
    with live:
        config = load_config(path_to_config)
        # set a seed so that random calculations are deterministic
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        console.log(f"Output folder is {data_output_folder}")
        assert os.path.exists(
            data_output_folder
        ), "Given data save folder does not exist"

        # load scenario info
        scenario_info = load_scenario_info_from_json(ScenarioInfo, base_data_folder)

        # put together start and goal point
        path_start = GlobalCoord(path_start_lon, path_start_lat)
        path_goal = GlobalCoord(path_goal_lon, path_goal_lat)

        start_execution_time = time.time()

        # generate output folder path
        dto = dt.now()  # datetime object

        mopp_folder_name = (
            f"MOPP_"
            f""
            f"{get_mopp_identifier(scenario_info, path_start, path_goal)}_"
            f"{dto.strftime('%y_%m_%d_%H_%M_%S')}"
        )

        mopp_output_path = os.path.join(data_output_folder, mopp_folder_name)
        # Reset singleton instance before creating a new one,
        # important when mopp.py is used several times in multiprocessing Pool
        DataManager.reset_instance()
        DataManager(
            input_path=data_input_folder,
            processing_path=data_processing_folder,
            output_path=mopp_output_path,
        )

        # Generate scenario that will consist of information about the city model,
        # noise, risk and nfa maps and a demand on the logistics network

        # first initialization of Scenario (Singleton) with scenario info
        # Reset singleton instance before creating a new one,
        # important when mopp.py is used several times in multiprocessing Pool
        Scenario.reset_instance()
        Scenario(scenario_info)

        # create path factory
        # Reset singleton instance before creating a new one,
        # important when mopp.py is used several times in multiprocessing Pool
        PathFactory.reset_instance()
        path_factory = PathFactory(
            input_data_folder=data_input_folder,
            global_start_pos=path_start,
            global_goal_pos=path_goal,
        )

        evaluator = Evaluator()

        for constraint_str in config.constraints:
            constraint_checker_module_str = config.get(
                constraint_str
            ).constraint_checker_module
            constraint_checker_class_str = config.get(
                constraint_str
            ).constraint_checker_class
            inputs_str = config.get(constraint_str).inputs
            parameters = config.get(constraint_str).parameters
            child_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "constraint_checkers")
            )
            sys.path.append(child_dir)
            try:
                map_creator_module = __import__(constraint_checker_module_str)
            except ModuleNotFoundError:
                assert_msg = (
                    f"No module {constraint_checker_module_str} found in "
                    f"folder 'constraint_checkers'"
                )
                raise AssertionError(assert_msg) from None
            constraint_checker_class = getattr(
                map_creator_module, constraint_checker_class_str
            )
            constraint_checker = constraint_checker_class(
                identifier_str=constraint_str, parameters=parameters
            )
            do_instance_check(constraint_checker)
            constraint_checker.load_inputs_and_init(inputs_str, data_input_folder)
            evaluator.add_constraint_checker(constraint_checker)

        # apply correct simulators according to objectives
        for objective_str in config.objectives:
            simulator_module_str = config.get(objective_str).simulator_module
            simulator_class_str = config.get(objective_str).simulator_class
            inputs_str = config.get(objective_str).inputs
            parameters = config.get(objective_str).parameters
            child_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "simulators")
            )
            sys.path.append(child_dir)
            try:
                simulator_module = __import__(simulator_module_str)
            except ModuleNotFoundError:
                assert_msg = (
                    f"No module {simulator_module_str} found in " f"folder 'simulators'"
                )
                raise AssertionError(assert_msg) from Exception
            simulator_class = getattr(simulator_module, simulator_class_str)
            simulator = simulator_class(
                identifier_str=objective_str, parameters=parameters
            )
            do_instance_check(simulator)
            simulator.load_inputs_and_init(inputs_str, data_input_folder)
            evaluator.add_simulator(simulator)

        # initialize preprocessor
        start_time = time.time()
        pre_processor = PreProcessor(evaluator)

        initial_paths = pre_processor.calculate_initial_paths()
        prep_calc_time = time.time() - start_time
        if kwargs.get("only_graph_creation", False):
            console.log("Finishing mopp after graph creation")
            sys.exit()

        # get optimizer properties
        optimizer_module_str = config.optimizer.optimizer_module
        optimizer_class_str = config.optimizer.optimizer_class
        optimizer_identifier = config.optimizer.optimizer_identifier
        optimizer_parameters = config.optimizer.parameters

        # initialize optimizer
        child_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "optimizers")
        )
        sys.path.append(child_dir)
        try:
            optimizer_module = __import__(optimizer_module_str)
        except ModuleNotFoundError:
            assert_msg = (
                f"No module {optimizer_module_str} found in " f"folder 'optimizers'"
            )
            raise AssertionError(assert_msg) from None
        optimizer_class = getattr(optimizer_module, optimizer_class_str)
        optimizer = optimizer_class(
            data_output_folder=mopp_output_path,
            optimizer_identifier=optimizer_identifier,
            optimizer_parameters=optimizer_parameters,
            evaluator=evaluator,
            initial_paths=initial_paths,
        )

        # start the optimization
        try:
            optimizer.init_optimizer()
            try:
                optimizer.optimize()
            except KeyboardInterrupt:
                console.log("-- End due to Keyboard interrupt --")
            optimizer.close_optimization()

        except NoSuitableApproximation:
            end_execution_time = time.time()
            console.log(
                f"Partial program execution took "
                f"{end_execution_time - start_execution_time} seconds."
            )
            console.log(
                "Did not find suitable NURBS approximation for a Dijkstra path "
                "in one of the programs. Will stop program."
            )
            sys.exit()
        except OptimizationEndsInNiches:
            end_execution_time = time.time()
            console.log(
                f"Partial program execution took "
                f"{end_execution_time - start_execution_time} seconds."
            )
            console.log(
                "Could not resolve Niches in one of the programs. Will stop program."
            )
            sys.exit()

        optimizer.finish_after_optimization()
        end_execution_time = time.time()

        save_mopp_statistics(optimizer, pre_processor, path_factory, prep_calc_time)

        console.log(
            f"Total program execution took "
            f"{end_execution_time - start_execution_time} seconds. "
        )


def save_mopp_statistics(optimizer, pre_processor, path_factory, prep_calc_time):
    mopp_statistics = {}
    mopp_statistics["opt_num_cp"] = int(path_factory.num_control_points)
    mopp_statistics["nurbs_order"] = int(path_factory.nurbs_order)
    mopp_statistics["global_start_pos"] = path_factory.global_start_pos
    mopp_statistics["global_goal_pos"] = path_factory.global_goal_pos
    mopp_statistics["default_num_graph_edges"] = pre_processor.default_num_graph_edges
    mopp_statistics["default_num_graph_nodes"] = pre_processor.default_num_graph_nodes
    mopp_statistics["clipped_num_graph_edges"] = pre_processor.clipped_num_graph_edges
    mopp_statistics["clipped_num_graph_nodes"] = pre_processor.clipped_num_graph_nodes
    mopp_statistics["num_eval"] = optimizer.eval_count
    mopp_statistics["num_gen"] = optimizer.generation_count
    mopp_statistics["raw_calc_time"] = optimizer.raw_calc_time
    mopp_statistics["eval_calc_time"] = optimizer.eval_calc_time
    mopp_statistics["prep_calc_time"] = prep_calc_time

    # Save the dictionary to a JSON file
    mopp_statistics["global_start_pos"] = asdict(mopp_statistics["global_start_pos"])
    mopp_statistics["global_goal_pos"] = asdict(mopp_statistics["global_goal_pos"])
    with open(
        os.path.join(DataManager().data_output_path, "mopp_statistics.json"), "w"
    ) as json_file:
        json.dump(
            mopp_statistics, json_file, indent=4
        )  # 'indent=4' adds indentation for readability


def load_mopp_statistics(save_folder: str):
    with open(os.path.join(save_folder, "mopp_statistics.json")) as json_file:
        mopp_statistics = json.load(json_file)

    # Convert the 'person' back to a Person dataclass instance
    mopp_statistics["global_start_pos"] = GlobalCoord(
        **mopp_statistics["global_start_pos"]
    )
    mopp_statistics["global_goal_pos"] = GlobalCoord(
        **mopp_statistics["global_goal_pos"]
    )

    return mopp_statistics


def do_instance_check(object_to_check):
    if not isinstance(object_to_check, Simulator | ConstraintChecker):
        raise TypeError("Object must be either a Simulator or a ConstraintChecker")
    if not isinstance(object_to_check, GraphBased | NonGraphBased):
        raise TypeError("Object must be either GraphBased or NonGraphBased")


if __name__ == "__main__":
    base_folder = sys.argv[1]
    assert os.path.isdir(base_folder), "No existing base data folder"

    path_start_lon = float(sys.argv[2])
    path_start_lat = float(sys.argv[3])
    path_goal_lon = float(sys.argv[4])
    path_goal_lat = float(sys.argv[5])

    path_config = sys.argv[6]
    assert os.path.exists(path_config), "No valid path to config"
    input_folder = sys.argv[7]
    assert os.path.isdir(input_folder), "No valid path to input folder"
    data_output_folder = sys.argv[8]
    assert os.path.isdir(data_output_folder), "No valid data save folder"

    data_proc_folder = sys.argv[9]
    assert os.path.isdir(data_proc_folder), "No valid data processing folder"
    kwargs = {}
    if len(sys.argv) > 10:
        args = sys.argv[10:]
        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                kwargs[key] = value
            else:
                print(f"Invalid keyword argument: {arg}")
    else:
        kwargs = {}
    main(
        base_folder,
        path_start_lon,
        path_start_lat,
        path_goal_lon,
        path_goal_lat,
        path_config,
        input_folder,
        data_output_folder,
        data_proc_folder,
        **kwargs,
    )
