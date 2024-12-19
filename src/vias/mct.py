# mct = Map Creation tool
import os
import pickle
import sys

from vias.config import get_config, load_config
from vias.console_manager import console
from vias.grid_graph import get_grid_graph_path, load_grid_graph, save_grid_graph
from vias.grid_map import GridMap
from vias.met import get_grid_map_plots_path, get_radio_tower_path
from vias.utils.helpers import (
    ScenarioInfo,
    get_map_identifier,
    load_scenario_info_from_json,
)


def load_map_creation_inputs(
    inputs_str: list[str], data_input_folder: str, scenario_info: ScenarioInfo
) -> dict:
    """
    By default, the inputs are searched in the "grid_maps" folder within the
    data_input_folder
    """
    inputs = {}
    config = get_config()
    for input_str in inputs_str:
        if "map" in input_str:
            map_dim = config.get(input_str).dimension
            path_to_input = os.path.join(
                data_input_folder,
                "grid_maps",
                f""
                f""
                f""
                f""
                f"{get_map_identifier(scenario_info, map_dim)}_{input_str}.pkl",
            )
        elif "radio" in input_str:
            path_to_input = get_radio_tower_path(data_input_folder, scenario_info)
        else:
            raise AssertionError(f"Unknown input_str {input_str}")
        assert os.path.exists(path_to_input), (
            f"Wanted input {path_to_input} does not exits. Please adjust path or make "
            f"sure it is created first."
        )
        with open(path_to_input, "rb") as f:
            input_data = pickle.load(f)
        inputs[input_str] = input_data
    return inputs


def start_grid_map_creation(
    data_input_folder: str, map_to_be_created: str, scenario_info: ScenarioInfo
) -> GridMap:
    config = get_config()
    map_creator_module_str = config.get(map_to_be_created).map_creator_module
    map_creator_class_str = config.get(map_to_be_created).map_creator_class
    inputs_str = config.get(map_to_be_created).inputs
    inputs = load_map_creation_inputs(inputs_str, data_input_folder, scenario_info)
    parameters = config.get(map_to_be_created).parameters
    child_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "map_creators"))
    sys.path.append(child_dir)
    try:
        map_creator_module = __import__(map_creator_module_str)
    except ModuleNotFoundError:
        assert_msg = (
            f"No module {map_creator_module_str} found in " f"folder 'map_creators'"
        )
        raise AssertionError(assert_msg) from None
    map_creator_class = getattr(map_creator_module, map_creator_class_str)
    map_creator = map_creator_class(inputs, scenario_info, parameters)
    grid_map = map_creator.create_map(map_to_be_created)
    return grid_map


def main(
    base_data_folder: str,
    path_to_config: str,
    data_input_folder: str,
    data_save_folder: str,
):
    """
    :param scenario_info:
    :param data_input_folder:
    :return:
    """
    config = load_config(path_to_config)

    assert os.path.exists(data_save_folder), "Given data save folder does not exist"

    # load scenario info
    scenario_info = load_scenario_info_from_json(ScenarioInfo, base_data_folder)

    for map_to_be_created in config.maps_to_be_created:
        # name of grid_map to be created
        output_str = config.get(map_to_be_created).output
        grid_map_path = get_grid_graph_path(data_save_folder, scenario_info, output_str)
        if os.path.exists(grid_map_path):
            console.log(
                f"Grid grid_map {grid_map_path} already exists. Skip calculation."
            )
            grid_map = load_grid_graph(data_input_folder, scenario_info, output_str)
            assert isinstance(grid_map, GridMap)
        else:
            grid_map = start_grid_map_creation(
                data_input_folder, map_to_be_created, scenario_info
            )

        # plot grid map
        console.log("Plotting grid map")
        grid_map_plots_path = get_grid_map_plots_path(data_save_folder)
        grid_map.plot_layer_flat(
            savepath=os.path.join(
                grid_map_plots_path,
                f"{get_map_identifier(scenario_info, grid_map.dimension)}_"
                f"{grid_map.name}",
            )
        )
        if grid_map.dimension == 3:
            grid_map.plot_volume(
                save_path=os.path.join(
                    grid_map_plots_path,
                    f"{get_map_identifier(scenario_info, grid_map.dimension)}_"
                    f"{grid_map.name}_volume",
                )
            )
            grid_map.plot_slices(grid_map_plots_path)
        # save grid map
        console.log("Saving grid map")
        save_grid_graph(grid_map, data_save_folder, scenario_info, output_str)


if __name__ == "__main__":
    base_folder = sys.argv[1]
    assert os.path.isdir(base_folder), "No existing base data folder"
    path_config = sys.argv[2]
    assert os.path.exists(path_config), "No valid path to config"
    input_folder = sys.argv[3]
    assert os.path.isdir(input_folder), "No valid path to input folder"
    data_output_folder = sys.argv[4]
    assert os.path.isdir(data_output_folder), "No valid data save folder"

    main(base_folder, path_config, input_folder, data_output_folder)
