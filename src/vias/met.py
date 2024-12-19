# met == Map Extraction Tool
import copy as cp
import os
import pickle
import sys
from typing import List

import numpy as np
import shapely
from affine import Affine
import matplotlib.colors as mpl_col
from matplotlib import pyplot as plt
from rasterio.features import rasterize

from vias.config import load_config, get_config
from vias.grid_graph import load_grid_graph, get_grid_graph_path
from src.vias.grid_map import GridMap
from vias.ocid_exporter import OcidExporter
from src.vias.osm_exporter import OsmExporter, CityModel
from vias.utils.helpers import ScenarioInfo, get_osm_identifier, get_map_identifier, get_tmerc_map_origin
from vias.utils.tools import get_polygon_area, euclidian_distance
from vias.console_manager import console

def get_semantic_geo_value_pairs(objects_to_render, scenario_info):
    shapes = []
    # first the coastline information has to be extracted to distinguish landmass from water (ocean)
    coastlines = {k: v for k, v in objects_to_render.items() if v["color_tag"] == 6.4}
    coast_line_polygon, other_polygons = coastline2polygon(coastlines, scenario_info.x_length, scenario_info.y_length)
    if len(coast_line_polygon) > 0:
        shapes.append((shapely.Polygon(coast_line_polygon), 6.4))
        for polygon in other_polygons:
            shapes.append((shapely.Polygon(polygon), 6.4))
    objects_to_render = {k: v for k, v in objects_to_render.items() if k not in coastlines.keys()}
    sorted_rendered_objects = {k: v for k, v in sorted(objects_to_render.items(),
                                                       key=lambda item: get_polygon_area(item[1]['pos']),
                                                       reverse=True)}
    for idx, rendered_object in enumerate(sorted_rendered_objects.values()):
        if len(rendered_object['pos']) <= 2:
            shapes.append((shapely.LineString(rendered_object['pos']), rendered_object['color_tag']))
        else:
            shapes.append((shapely.Polygon(rendered_object['pos']), rendered_object['color_tag']))
    return shapes


def get_street_geo_value_pairs(objects_to_render):
    shapes = []
    for idx, rendered_object in enumerate(objects_to_render.values()):
        if len(rendered_object['pos']) < 2:
            continue
        shapes.append((shapely.LineString(rendered_object['pos']), rendered_object['color_tag']))
    return shapes


def get_building_geo_value_pairs(objects_to_render):
    shapes = []
    sorted_rendered_objects = {k: v for k, v in sorted(objects_to_render.items(),
                                                       key=lambda item: get_polygon_area(item[1]['pos']),
                                                       reverse=True)}
    for idx, rendered_object in enumerate(sorted_rendered_objects.values()):
        if len(rendered_object['pos']) <= 2:
            shapes.append((shapely.LineString(rendered_object['pos']), rendered_object['height']))
        else:
            shapes.append((shapely.Polygon(rendered_object['pos']), rendered_object['height']))
    return shapes

def get_radio_tower_geo_value_pairs(objects_to_render):
    shapes = []
    for tower_pos in objects_to_render:
        shapes.append((shapely.Point(tower_pos), 1.0))
    return shapes

def extract_city_model(scenario_info: ScenarioInfo, path_to_osm_file: str,
                       data_save_folder: str):
    city_model_path = get_city_model_path(data_save_folder, scenario_info)
    console.log(f"Looking for city model file in {city_model_path}")
    if not os.path.exists(city_model_path):
        console.log(f"Does not exist. Create it.")
        osm_exporter = OsmExporter(path_to_osm_file, data_save_folder)

        city_model = osm_exporter.calculate_city_model(scenario_info)

        with open(city_model_path, 'wb') as handle:
            pickle.dump(city_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"City model {city_model_path} already exists")

def extract_ocid_radio_towers(scenario_info: ScenarioInfo, path_to_ocid_files: str,
                       data_save_folder: str):
    radio_tower_path = get_radio_tower_path(data_save_folder, scenario_info)

    if not os.path.exists(radio_tower_path):
        ocid_exporter = OcidExporter(path_to_ocid_files, data_save_folder)
        radio_tower_positions = ocid_exporter.get_cell_positions(scenario_info)
        with open(radio_tower_path, 'wb') as handle:
            pickle.dump(radio_tower_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Radio positions file {radio_tower_path} already exists")

def load_city_model(scenario_info: ScenarioInfo, data_save_folder: str) -> CityModel:
    city_model_path = get_city_model_path(data_save_folder, scenario_info)
    if os.path.exists(city_model_path):
        city_model = pickle.load(open(city_model_path, 'rb'))
        return city_model
    else:
        raise Exception("met.py", f"Tried to load city model {city_model_path} that does not exist.")

def load_radio_towers(scenario_info: ScenarioInfo, data_save_folder: str) -> List:
    radio_tower_path = get_radio_tower_path(data_save_folder, scenario_info)
    if os.path.exists(radio_tower_path):
        radio_towers = pickle.load(open(radio_tower_path, 'rb'))
        return radio_towers
    else:
        raise Exception("met.py", f"Tried to load radio towers {radio_tower_path} that does not exist.")

def get_city_model_path(data_save_folder, scenario_info):
    path_to_city_models = os.path.join(data_save_folder, 'city_models')
    if not os.path.exists(path_to_city_models):
        os.makedirs(path_to_city_models)

    city_model_name = get_osm_identifier(scenario_info)
    city_model_path = os.path.join(path_to_city_models, f'{city_model_name}_city_model.pkl')
    return city_model_path

def get_radio_tower_path(data_save_folder, scenario_info):
    path_to_radio_towers = os.path.join(data_save_folder, 'radio_towers')
    if not os.path.exists(path_to_radio_towers):
        os.makedirs(path_to_radio_towers)

    radio_tower_name = get_osm_identifier(scenario_info)
    radio_tower_path = os.path.join(path_to_radio_towers, f'{radio_tower_name}_radio_towers.pkl')
    return radio_tower_path

def get_rasterio_transform(scenario_info: ScenarioInfo):
    """Calculates an affine transform
        | x' |   | a  b  c | | x |
        | y' | = | d  e  f | | y |
        | 1  |   | g  h  i | | 1 |

        whereas x' and y' are the spatial (e.g. tmerc) coordinates with x' (=east) pointing to the right and y' (=north) pointing
        upwards;

        and x and y are the raster coordinates of the "image" with its coordinate frame located in the upper left
        x (column index) pointing to the right, and y (row index) pointing downwards.

        """
    tmerc_map_origin = get_tmerc_map_origin(scenario_info)
    T_trans = np.array([[1, 0, tmerc_map_origin.east], [0, 1, tmerc_map_origin.north], [0, 0, 1]])
    T_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T_scale = np.array([[scenario_info.x_res, 0, 0], [0, scenario_info.y_res, 0], [0, 0, 1]])

    T_tot = np.matmul(np.matmul(T_trans, T_rot), T_scale)

    return Affine(*T_tot.flatten()[0:6])


def extract_grid_maps(scenario_info: ScenarioInfo, data_save_folder: str, no_ocid: bool):
    config = get_config()
    city_model = load_city_model(scenario_info, data_save_folder)
    radio_towers = load_radio_towers(scenario_info, data_save_folder)
    rasterio_transform = get_rasterio_transform(scenario_info)

    grid_map_plots_path = get_grid_map_plots_path(data_save_folder)

    street_shapes = []
    building_shapes = []
    maps_to_extract = ['buildings_map', 'streets_map', 'semantic_map']
    if not no_ocid:
        maps_to_extract.append('radio_signal_towers_map')
    for map_name in maps_to_extract:
        map_save_path = get_grid_graph_path(data_save_folder, scenario_info, map_name)

        if not os.path.exists(map_save_path):
            console.log("Map {} is being rendered. This may take a while ...".format(map_name))

            if map_name == "semantic_map":
                init_val = 6.4  # defaut value is 6.4 for coastline (landmass)
                shapes = get_semantic_geo_value_pairs(city_model.semantics, scenario_info)
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
            grid_map = GridMap(scenario_info.convert_to_map_info(map_name), dimension=map_dim, init_val=init_val)

            rendered_grid_array = grid_map.grid_array
            if map_name == "semantic_map":
                building_shapes = [(el[0], 172.8) for el in building_shapes]
                street_shapes = [(el[0], 121.6) for el in street_shapes]
                if len(building_shapes) > 0:
                    rendered_grid_array = rasterize(building_shapes, out=rendered_grid_array, transform=rasterio_transform,
                                                    all_touched=True)
                if len(street_shapes) > 0:
                    rendered_grid_array = rasterize(street_shapes, out=rendered_grid_array, transform=rasterio_transform,
                                                    all_touched=True)

            if len(shapes) > 0:
                rendered_grid_array = rasterize(shapes, out=rendered_grid_array, transform=rasterio_transform,
                                                all_touched=True)

            # do some checks
            if map_name == "buildings_map":
                rendered_grid_array[rendered_grid_array < 0.0] = 0.0  # set eventual negative values to zero

            if map_name == "streets_map":
                rendered_grid_array[rendered_grid_array > 0.0]  = 1.0
                rendered_grid_array[rendered_grid_array <= 0.0] = 0.0



            grid_map.set_from_array(rendered_grid_array)

            with open(map_save_path, 'wb') as handle:
                pickle.dump(grid_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f"Map {map_name} already exists. Skip calculation. Only plot.")
            grid_map = load_grid_graph(data_save_folder, scenario_info, map_name)

        if "semantic" in grid_map.name:
            tab20 = plt.get_cmap('tab20')

            # Step 2: Convert the colormap to a list of hex values
            tab20_colors = [tab20(i) for i in range(tab20.N)]  # Extract RGB colors
            tab20_colors_hex = [mpl_col.rgb2hex(c) for c in tab20_colors]  # Convert to hex

            # Step 3: Create a Plotly colorscale from the Matplotlib colormap
            # Normalize color scale range to [0, 1]
            plotly_colorscale = [[i / (len(tab20_colors_hex) - 1), color] for i, color in
                                 enumerate(tab20_colors_hex)]
            grid_map.plot_layer_flat(colormap=plotly_colorscale, norm_min_max=(0, 255),
                                     savepath=os.path.join(grid_map_plots_path,
                                                           f"{get_map_identifier(scenario_info, grid_map.dimension)}_{map_name}"))
        else:
            grid_map.plot_layer_flat(savepath=os.path.join(grid_map_plots_path,
                                                           f"{get_map_identifier(scenario_info, grid_map.dimension)}_{map_name}"))



def get_grid_map_plots_path(data_save_folder):
    grid_map_plots_path = os.path.join(data_save_folder, "grid_map_plots")
    if not os.path.exists(grid_map_plots_path):
        os.makedirs(grid_map_plots_path)
    return grid_map_plots_path

def get_grid_map_folder_path(data_save_folder):
    grid_maps_path = os.path.join(data_save_folder, "grid_maps")
    if not os.path.exists(grid_maps_path):
        os.makedirs(grid_maps_path)
    return grid_maps_path



def main(path_to_osm_file: str, scenario_info: ScenarioInfo, path_to_config: str, data_save_folder: str, path_to_ocid_files=None):
    """
    :param scenario_info:
    :param data_input_folder:
    :return:
    """
    load_config(path_to_config)

    assert os.path.exists(data_save_folder), "Given data save folder does not exist"

    extract_city_model(scenario_info, path_to_osm_file, data_save_folder)

    no_ocid = True
    if path_to_ocid_files is not None:
        extract_ocid_radio_towers(scenario_info, path_to_ocid_files, data_save_folder)
        no_ocid = False

    extract_grid_maps(scenario_info, data_save_folder, no_ocid)




if __name__ == "__main__":
    main(*sys.argv[1:])


def coastline2polygon(coastlines: dict, x_length, y_length):
    if len(coastlines) == 0:
        return [], []
    coastlines = [val["npos"] for val in coastlines.values()]
    FUSE_DISTANCE = 5
    stacked_coastlines = []
    while len(coastlines) > 0:
        stacked_coastline = coastlines.pop(0)
        while True:
            if len(coastlines) == 0:
                break
            # distances from end of line to all other lines
            distances_to_start = [euclidian_distance(stacked_coastline[-1], line[0]) for line in coastlines]
            # distances from beginning of line to all other lines
            distances_to_end = [euclidian_distance(stacked_coastline[0], line[-1]) for line in coastlines]
            min_dist_to_start = min(distances_to_start)
            min_dist_to_end = min(distances_to_end)
            if min_dist_to_start < min_dist_to_end:
                if min_dist_to_start < FUSE_DISTANCE:
                    idx = distances_to_start.index(min_dist_to_start)
                    stacked_coastline.extend(coastlines.pop(idx))
                else:
                    break
            else:
                if min_dist_to_end < FUSE_DISTANCE:
                    idx = distances_to_end.index(min_dist_to_end)
                    stacked_coastline = coastlines.pop(idx) + stacked_coastline
                else:
                    break

        stacked_coastlines.append(stacked_coastline)

    # we have now all different coastlines stacked together
    # we assume that there is one large main coastline, all other coastlines are assumed to be separate polygons (e.g. islands)
    coastline_length = [len(set(cst)) for cst in stacked_coastlines]
    max_idx = coastline_length.index(max(coastline_length))
    main_coastline = stacked_coastlines.pop(max_idx)

    # there are several cases where there has to be added points to the coastline
    start, goal = get_coastline_edges(main_coastline, x_length, y_length, 250)
    start_x = main_coastline[0][0]
    start_y = main_coastline[0][1]
    goal_x = main_coastline[-1][0]
    goal_y = main_coastline[-1][1]
    if start == "l" and goal == "l":
        if start_y > goal_y:
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
    if start == "b" and goal == "b":
        if start_x < goal_x:
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
    if start == "t" and goal == "t":
        if start_x > goal_x:
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
            main_coastline.append((x_length, y_length))
    if start == "r" and goal == "r":
        if start_y < goal_y:
            main_coastline.append((x_length, y_length))
            main_coastline.append((0, y_length))
            main_coastline.append((0, 0))
            main_coastline.append((x_length, 0))
    if start == "l" and goal == "t":
        main_coastline.append((0, y_length))
    if start == "l" and goal == "r":
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
    if start == "l" and goal == "b":
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
    if start == "b" and goal == "l":
        main_coastline.append((0, 0))
    if start == "b" and goal == "t":
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
    if start == "b" and goal == "r":
        main_coastline.append((x_length, y_length))
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
    if start == "r" and goal == "t":
        main_coastline.append((0, y_length))
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
    if start == "r" and goal == "l":
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
    if start == "r" and goal == "b":
        main_coastline.append((x_length, 0))
    if start == "t" and goal == "l":
        main_coastline.append((0, 0))
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
    if start == "t" and goal == "b":
        main_coastline.append((x_length, 0))
        main_coastline.append((x_length, y_length))
    if start == "t" and goal == "r":
        main_coastline.append((x_length, y_length))

    return main_coastline, stacked_coastlines


def get_coastline_edges(coastline, x_length, y_length, threshold):
    """Returns a combination of "l", "r", "t", "b" (left, right, top, bottom) depending on the position of the first and the last
    point of the coastline dependent to the frame."""
    ret = []
    for p in [coastline[0], coastline[-1]]:
        if p[0] < 0 + threshold:
            ret.append("l")
        elif p[0] > x_length - threshold:
            ret.append("r")
        elif p[1] < 0 + threshold:
            ret.append("b")
        elif p[1] > y_length - threshold:
            ret.append("t")
        else:
            ret.append(-1)
    return tuple(ret)
