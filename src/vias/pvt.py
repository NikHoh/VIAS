# pvt == Path Visualization Tool
import os
import pickle
import sys

from vias.coder import Coder
from vias.config import get_config, load_config
from vias.console_manager import console
from vias.data_manager import DataManager
from vias.grid_graph import load_grid_graph
from vias.mopp import load_mopp_statistics
from vias.pareto_handler import ParetoHandler
from vias.path_factory import PathFactory
from vias.scenario import Scenario
from vias.utils.helpers import ScenarioInfo, load_scenario_info_from_json


def main(
    base_data_folder: str,
    path_to_config: str,
    path_to_optimization_results: str,
    path_to_grid_maps_folder="",
):
    load_config(path_to_config)
    assert os.path.exists(
        path_to_optimization_results
    ), "Given data save folder does not exist"

    DataManager(
        input_path="", processing_path="", output_path=path_to_optimization_results
    )

    # load scenario info
    scenario_info = load_scenario_info_from_json(ScenarioInfo, base_data_folder)

    mopp_statistics = load_mopp_statistics(DataManager().data_output_path)

    process_F_X_dict(
        path_to_optimization_results,
        path_to_grid_maps_folder,
        scenario_info,
        mopp_statistics,
    )

    process_optimizer_statistics(path_to_optimization_results)


def process_optimizer_statistics(data_save_folder):
    console.log("Loading optimization statistics")

    path_to_stats = os.path.join(data_save_folder, "optimizer_statistics.pkl")
    if not os.path.isfile(path_to_stats):
        console.log(f"No path {path_to_stats} found. Skip.")
        return
    pareto_handler = ParetoHandler(path_to_stats)

    pareto_handler.save_generation_statistic_plot()

    # here something with pareto_handler data can be done, e.g.

    console.print(f"The last ideal point was {pareto_handler.get_ideal_point()}.")


def process_F_X_dict(
    data_save_folder, path_to_grid_maps_folder, scenario_info, mopp_statistics: dict
):
    config = get_config()
    console.log("Loading F_X dict")
    path_to_F_X_dict = os.path.join(data_save_folder, "F_X_dict.pkl")
    if not os.path.isfile(path_to_F_X_dict):
        console.log(f"No path {path_to_F_X_dict} found. Skip.")
        return
    with open(path_to_F_X_dict, "rb") as f:
        F_X_dict = pickle.load(f)

    # plug data into ParetoHandler
    pareto_handler = ParetoHandler()
    pareto_handler.insert_F_X_dict(F_X_dict)

    # here something with the F_X data can be done, e.g.

    knee_point = pareto_handler.get_knee_point()
    console.print(f"Knee point is {knee_point}")

    if (
        scenario_info is not None
        and path_to_grid_maps_folder != ""
        and mopp_statistics is not None
    ):
        global_start_pos = mopp_statistics["global_start_pos"]
        global_goal_pos = mopp_statistics["global_goal_pos"]
        Scenario(scenario_info)
        path_factory = PathFactory(
            input_data_folder=path_to_grid_maps_folder,
            global_start_pos=global_start_pos,
            global_goal_pos=global_goal_pos,
            num_control_points=mopp_statistics["opt_num_cp"],
            nurbs_order=mopp_statistics["nurbs_order"],
        )

        coder = Coder()
        console.print("Retrieving knee point path")
        knee_point_path = coder.decode(pareto_handler.get_knee_point_path())
        console.print("Plotting path")
        grid_map = load_grid_graph(
            path_to_grid_maps_folder, scenario_info, "buildings_map"
        )
        if config.suppress_grid_image_plot:
            console.print(
                "Will not plot anything as grid_image_plot is disabled in config."
            )

        # insert take-off and landing sequence
        assert path_factory.takeoff_sequence is not None
        assert path_factory.landing_sequence is not None
        knee_point_path.insert_takoff_landing_sequence(
            path_factory.takeoff_sequence, path_factory.landing_sequence
        )

        grid_map.plot_3D_paths(
            knee_point_path,
            savepath=DataManager().data_output_path,
            prefix="3D_knee_path_",
        )
    else:
        console.print("Not all data available to print paths.")


if __name__ == "__main__":
    base_folder = sys.argv[1]
    assert os.path.isdir(base_folder), "No existing base data folder"
    path_config = sys.argv[2]
    assert os.path.exists(path_config), "No valid path to config"
    result_folder = sys.argv[3]
    assert os.path.isdir(result_folder), "No valid path to result/output folder"
    if len(sys.argv) > 4:
        grid_map_folder = sys.argv[4]
        assert os.path.isdir(grid_map_folder), "No valid grid map folder"
    else:
        grid_map_folder = ""

    main(
        base_folder,
        path_config,
        result_folder,
        path_to_grid_maps_folder=grid_map_folder,
    )
