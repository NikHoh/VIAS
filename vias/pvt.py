# pvt == Path Visualization Tool
import os
import pickle

from vias.coder import Coder
from vias.config import load_config, get_config
from vias.console_manager import console
from vias.data_manager import DataManager
from vias.grid_graph import load_grid_graph
from vias.mopp import load_mopp_statistics
from vias.pareto_handler import ParetoHandler
from vias.path_factory import PathFactory
from vias.scenario import Scenario
from vias.utils.helpers import ScenarioInfo
from vias.utils.helpers import load_scneario_info_from_json


def main(path_to_config: str, path_to_optimization_results: str, path_to_grid_maps_folder=""):
    config = load_config(path_to_config)
    assert os.path.exists(path_to_optimization_results), "Given data save folder does not exist"

    DataManager(input_path="", processing_path="", output_path=path_to_optimization_results)

    scenario_info = load_scneario_info_from_json(ScenarioInfo, DataManager().data_output_path)

    mopp_statistics = load_mopp_statistics(DataManager().data_output_path)

    process_F_X_dict(path_to_optimization_results, path_to_grid_maps_folder, scenario_info, mopp_statistics)

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


def process_F_X_dict(data_save_folder, path_to_grid_maps_folder, scenario_info, mopp_statistics:dict):
    config = get_config()
    console.log("Loading F_X dict")
    path_to_F_X_dict = os.path.join(data_save_folder, "F_X_dict.pkl")
    if not os.path.isfile(path_to_F_X_dict):
        console.log(f"No path {path_to_F_X_dict} found. Skip.")
        return
    with open(path_to_F_X_dict, 'rb') as f:
        F_X_dict = pickle.load(f)

    # plug data into ParetoHandler
    pareto_handler = ParetoHandler()
    pareto_handler.insert_F_X_dict(F_X_dict)

    # here something with the F_X data can be done, e.g.

    knee_point = pareto_handler.get_knee_point()
    console.print(f"Knee point is {knee_point}")

    if scenario_info is not None and path_to_grid_maps_folder != "" and mopp_statistics is not None:
        global_start_pos = mopp_statistics["global_start_pos"]
        global_goal_pos = mopp_statistics["global_goal_pos"]
        Scenario(scenario_info)
        path_factory = PathFactory(input_data_folder=path_to_grid_maps_folder, global_start_pos=global_start_pos,
                    global_goal_pos=global_goal_pos)
        path_factory.num_control_points = mopp_statistics["opt_num_cp"]
        coder = Coder()
        console.print("Retrieving knee point path")
        knee_point_path = coder.decode(pareto_handler.get_knee_point_path())
        console.print("Plotting path")
        grid_map = load_grid_graph(path_to_grid_maps_folder, scenario_info, "buildings_map")
        if config.suppress_grid_image_plot:
            console.print("Will not plot anything as grid_image_plot is disabled in config.")
        grid_map.plot_3D_paths(knee_point_path)
    else:
        console.print("Not all data available to print paths.")


if __name__ == "__main__":
    main()
