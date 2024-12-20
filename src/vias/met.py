# met == Map Extraction Tool
import copy as cp
import os
import pickle
import sys

import matplotlib.colors as mpl_col
import numpy as np
import shapely
from affine import Affine
from matplotlib import pyplot as plt
from rasterio.features import rasterize

from vias.config import get_config, load_config
from vias.console_manager import console
from vias.grid_graph import (
    get_grid_graph_path,
    load_grid_graph,
    save_grid_graph,
)
from vias.grid_map import GridMap
from vias.ocid_exporter import OcidExporter
from vias.osm_exporter import CityModel, OsmExporter
from vias.scenario import Scenario, get_tmerc_map_origin
from vias.utils.helpers import (
    ScenarioInfo,
    coastline2polygon,
    get_map_identifier,
    get_osm_identifier,
    load_scenario_info_from_json,
)
from vias.utils.tools import get_polygon_area


def get_semantic_geo_value_pairs(objects_to_render, scenario_info):
    shapes = []
    # first the coastline information has to be extracted to distinguish landmass
    # from water (ocean)
    coastlines = {k: v for k, v in objects_to_render.items() if v["color_tag"] == 6.4}
    coast_line_polygon, other_polygons = coastline2polygon(
        coastlines, scenario_info.x_length, scenario_info.y_length
    )
    if len(coast_line_polygon) > 0:
        shapes.append((shapely.Polygon(coast_line_polygon), 6.4))
        for polygon in other_polygons:
            shapes.append((shapely.Polygon(polygon), 6.4))
    objects_to_render = {
        k: v for k, v in objects_to_render.items() if k not in coastlines
    }
    sorted_rendered_objects = {
        k: v
        for k, v in sorted(
            objects_to_render.items(),
            key=lambda item: get_polygon_area(item[1]["pos"]),
            reverse=True,
        )
    }
    for rendered_object in sorted_rendered_objects.values():
        if len(rendered_object["pos"]) <= 2:
            shapes.append(
                (
                    shapely.LineString(rendered_object["pos"]),
                    rendered_object["color_tag"],
                )
            )
        else:
            shapes.append(
                (shapely.Polygon(rendered_object["pos"]), rendered_object["color_tag"])
            )
    return shapes


def get_street_geo_value_pairs(objects_to_render):
    shapes = []
    for rendered_object in objects_to_render.values():
        if len(rendered_object["pos"]) < 2:
            continue
        shapes.append(
            (shapely.LineString(rendered_object["pos"]), rendered_object["color_tag"])
        )
    return shapes


def get_building_geo_value_pairs(objects_to_render):
    shapes = []
    sorted_rendered_objects = {
        k: v
        for k, v in sorted(
            objects_to_render.items(),
            key=lambda item: get_polygon_area(item[1]["pos"]),
            reverse=True,
        )
    }
    for rendered_object in sorted_rendered_objects.values():
        if len(rendered_object["pos"]) <= 2:
            shapes.append(
                (shapely.LineString(rendered_object["pos"]), rendered_object["height"])
            )
        else:
            shapes.append(
                (shapely.Polygon(rendered_object["pos"]), rendered_object["height"])
            )
    return shapes


def get_radio_tower_geo_value_pairs(objects_to_render):
    shapes = []
    for tower_pos in objects_to_render:
        shapes.append((shapely.Point(tower_pos), 1.0))
    return shapes


def extract_city_model(
    scenario_info: ScenarioInfo, path_to_osm_file: str, data_save_folder: str
):
    city_model_path = get_city_model_path(data_save_folder, scenario_info)
    console.log(f"Looking for city model file in {city_model_path}")
    if not os.path.exists(city_model_path):
        console.log("Does not exist. Create it.")
        osm_exporter = OsmExporter(path_to_osm_file, data_save_folder)

        city_model = osm_exporter.calculate_city_model(scenario_info)

        with open(city_model_path, "wb") as handle:
            pickle.dump(city_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"City model {city_model_path} already exists")


def extract_ocid_radio_towers(
    scenario_info: ScenarioInfo, path_to_ocid_files: str, data_save_folder: str
):
    radio_tower_path = get_radio_tower_path(data_save_folder, scenario_info)

    if not os.path.exists(radio_tower_path):
        ocid_exporter = OcidExporter(path_to_ocid_files, data_save_folder)
        radio_tower_positions = ocid_exporter.get_cell_positions(scenario_info)
        with open(radio_tower_path, "wb") as handle:
            pickle.dump(radio_tower_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Radio positions file {radio_tower_path} already exists")


def load_city_model(scenario_info: ScenarioInfo, data_save_folder: str) -> CityModel:
    city_model_path = get_city_model_path(data_save_folder, scenario_info)
    if os.path.exists(city_model_path):
        with open(city_model_path, "rb") as f:
            city_model = pickle.load(f)
        return city_model
    else:
        raise Exception(
            "met.py",
            f"Tried to load city model {city_model_path} that does not " f"exist.",
        )


def load_radio_towers(scenario_info: ScenarioInfo, data_save_folder: str) -> list:
    radio_tower_path = get_radio_tower_path(data_save_folder, scenario_info)
    if os.path.exists(radio_tower_path):
        with open(radio_tower_path, "rb") as f:
            radio_towers = pickle.load(f)
        return radio_towers
    else:
        raise Exception(
            "met.py",
            f"Tried to load radio towers {radio_tower_path} that does not " f"exist.",
        )


def get_city_model_path(data_save_folder, scenario_info):
    path_to_city_models = os.path.join(data_save_folder, "city_models")
    if not os.path.exists(path_to_city_models):
        os.makedirs(path_to_city_models)

    city_model_name = get_osm_identifier(scenario_info)
    city_model_path = os.path.join(
        path_to_city_models, f"{city_model_name}_city_model.pkl"
    )
    return city_model_path


def get_radio_tower_path(data_save_folder, scenario_info):
    path_to_radio_towers = os.path.join(data_save_folder, "radio_towers")
    if not os.path.exists(path_to_radio_towers):
        os.makedirs(path_to_radio_towers)

    radio_tower_name = get_osm_identifier(scenario_info)
    radio_tower_path = os.path.join(
        path_to_radio_towers, f"{radio_tower_name}_radio_towers.pkl"
    )
    return radio_tower_path


def get_rasterio_transform(scenario_info: ScenarioInfo):
    """Calculates an affine transform
    | x' |   | a  b  c | | x |
    | y' | = | d  e  f | | y |
    | 1  |   | g  h  i | | 1 |

    whereas x' and y' are the spatial (e.g. tmerc) coordinates with x' (=east) pointing
    to the right and y' (=north) pointing upwards;

    and x and y are the raster coordinates of the "image" with its coordinate frame
    located in the upper left x (column index) pointing to the right, and y (row index)
    pointing downwards.

    """
    tmerc_map_origin = get_tmerc_map_origin(scenario_info)
    T_trans = np.array(
        [[1, 0, tmerc_map_origin.east], [0, 1, tmerc_map_origin.north], [0, 0, 1]]
    )
    T_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T_scale = np.array(
        [[scenario_info.x_res, 0, 0], [0, scenario_info.y_res, 0], [0, 0, 1]]
    )

    T_tot = np.matmul(np.matmul(T_trans, T_rot), T_scale)

    return Affine(*T_tot.flatten()[0:6])


def extract_grid_maps(
    scenario_info: ScenarioInfo, data_save_folder: str, no_ocid: bool
):
    config = get_config()
    city_model = load_city_model(scenario_info, data_save_folder)

    rasterio_transform = get_rasterio_transform(scenario_info)

    grid_map_plots_path = get_grid_map_plots_path(data_save_folder)

    street_shapes = []
    building_shapes = []
    maps_to_extract = ["buildings_map", "streets_map", "semantic_map"]
    if not no_ocid:
        maps_to_extract.append("radio_signal_towers_map")
        radio_towers = load_radio_towers(scenario_info, data_save_folder)
    for map_name in maps_to_extract:
        map_save_path = get_grid_graph_path(data_save_folder, scenario_info, map_name)

        if not os.path.exists(map_save_path):
            console.log(f"Map {map_name} is being rendered. This may take a while ...")

            if map_name == "semantic_map":
                init_val = 6.4  # defaut value is 6.4 for coastline (landmass)
                shapes = get_semantic_geo_value_pairs(
                    city_model.semantics, scenario_info
                )
            else:
                init_val = 0
                if map_name == "buildings_map":
                    shapes = get_building_geo_value_pairs(city_model.buildings)
                    building_shapes = cp.deepcopy(shapes)
                elif map_name == "streets_map":
                    shapes = get_street_geo_value_pairs(city_model.streets)
                    street_shapes = cp.deepcopy(shapes)
                elif map_name == "radio_signal_towers_map":
                    shapes = get_radio_tower_geo_value_pairs(radio_towers)
                else:
                    raise Exception("Unknown grid_map type")
            map_dim = config.get(map_name).dimension
            grid_map: GridMap = GridMap(
                scenario_info.convert_to_map_info(map_name),
                dimension=map_dim,
                init_val=init_val,
            )

            rendered_grid_array = grid_map.grid_array
            if map_name == "semantic_map":
                building_shapes = [(el[0], 172.8) for el in building_shapes]
                street_shapes = [(el[0], 121.6) for el in street_shapes]
                if len(building_shapes) > 0:
                    rendered_grid_array = rasterize(
                        building_shapes,
                        out=rendered_grid_array,
                        transform=rasterio_transform,
                        all_touched=True,
                    )
                if len(street_shapes) > 0:
                    rendered_grid_array = rasterize(
                        street_shapes,
                        out=rendered_grid_array,
                        transform=rasterio_transform,
                        all_touched=True,
                    )

            if len(shapes) > 0:
                rendered_grid_array = rasterize(
                    shapes,
                    out=rendered_grid_array,
                    transform=rasterio_transform,
                    all_touched=True,
                )

            # do some checks
            if map_name == "buildings_map":
                rendered_grid_array[rendered_grid_array < 0.0] = (
                    0.0
                    # set eventual negative values to zero
                )

            if map_name == "streets_map":
                rendered_grid_array[rendered_grid_array > 0.0] = 1.0
                rendered_grid_array[rendered_grid_array <= 0.0] = 0.0

            grid_map.set_from_array(rendered_grid_array)

            save_grid_graph(grid_map, data_save_folder, scenario_info, map_name)
        else:
            print(f"Map {map_name} already exists. Skip calculation. Only plot.")
            loaded_map = load_grid_graph(data_save_folder, scenario_info, map_name)
            assert isinstance(loaded_map, GridMap)
            grid_map = loaded_map

        assert isinstance(grid_map, GridMap)
        if "semantic" in grid_map.name:
            tab20 = plt.get_cmap("tab20")

            # Step 2: Convert the colormap to a list of hex values
            tab20_colors = [tab20(i) for i in range(tab20.N)]  # Extract RGB colors
            tab20_colors_hex = [
                mpl_col.rgb2hex(c) for c in tab20_colors
            ]  # Convert to hex

            # Step 3: Create a Plotly colorscale from the Matplotlib colormap
            # Normalize color scale range to [0, 1]
            plotly_colorscale = [
                [i / (len(tab20_colors_hex) - 1), color]
                for i, color in enumerate(tab20_colors_hex)
            ]
            grid_map.plot_layer_flat(
                colormap=plotly_colorscale,
                norm_min_max=(0, 255),
                savepath=os.path.join(
                    grid_map_plots_path,
                    f"{get_map_identifier(scenario_info, grid_map.dimension)}_"
                    f"{map_name}",
                ),
            )
        else:
            grid_map.plot_layer_flat(
                savepath=os.path.join(
                    grid_map_plots_path,
                    f"{get_map_identifier(scenario_info, grid_map.dimension)}_"
                    f"{map_name}",
                )
            )


def get_grid_map_plots_path(data_save_folder):
    grid_map_plots_path = os.path.join(data_save_folder, "grid_map_plots")
    if not os.path.exists(grid_map_plots_path):
        os.makedirs(grid_map_plots_path)
    return grid_map_plots_path


def main(
    path_to_osm_file: str,
    base_data_folder: str,
    path_to_config: str,
    data_save_folder: str,
    path_to_ocid_files=None,
):
    """
    :param scenario_info:
    :param data_input_folder:
    :return:
    """

    load_config(path_to_config)

    # load scenario info
    scenario_info = load_scenario_info_from_json(ScenarioInfo, base_data_folder)
    Scenario.reset_instance()
    Scenario(scenario_info)

    assert os.path.exists(data_save_folder), "Given data save folder does not exist"

    extract_city_model(scenario_info, path_to_osm_file, data_save_folder)

    no_ocid = True
    if path_to_ocid_files is not None:
        extract_ocid_radio_towers(scenario_info, path_to_ocid_files, data_save_folder)
        no_ocid = False

    extract_grid_maps(scenario_info, data_save_folder, no_ocid)


if __name__ == "__main__":
    path_to_osm = sys.argv[1]
    assert os.path.exists(path_to_osm), "No valid path to OSM file"
    base_folder = sys.argv[2]
    assert os.path.isdir(base_folder), "No existing base data folder"
    path_config = sys.argv[3]
    assert os.path.exists(path_config), "No valid path to config"
    data_output_folder = sys.argv[4]
    assert os.path.isdir(data_output_folder), "No valid data save folder"
    path_to_ocid = None
    if len(sys.argv) == 6:
        path_to_ocid = sys.argv[5]
        assert os.path.isdir(path_to_ocid), "No valid path to OCID file"

    main(path_to_osm, base_folder, path_config, data_output_folder, path_to_ocid)
