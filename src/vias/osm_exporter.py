import os
import subprocess

from osmread import Node, Relation, Way, parse_file

from vias.utils.tools import euclidian_distance

from .utils.helpers import (
    GlobalCoord,
    ScenarioInfo,
    TmercCoord,
    get_osm_identifier,
    get_projection,
    get_tmerc_map_center,
    tmerc_coord_from_global_coord,
)


class NoNumericLiteralException(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def get_global_osm_bounding_box(scenario_info) -> tuple[GlobalCoord, GlobalCoord]:
    t_merc_map_center = get_tmerc_map_center(scenario_info)
    tmerc_bounding_box_west = (
        t_merc_map_center.east
        - int(scenario_info.x_length / 2)
        - int(scenario_info.x_length / 10)
    )
    tmerc_bounding_box_north = (
        t_merc_map_center.north
        + int(scenario_info.y_length / 2)
        + int(scenario_info.y_length / 10)
    )
    tmerc_bounding_box_east = (
        t_merc_map_center.east
        + int(scenario_info.x_length / 2)
        + int(scenario_info.x_length / 10)
    )
    tmerc_bounding_box_south = (
        t_merc_map_center.north
        - int(scenario_info.y_length / 2)
        - int(scenario_info.y_length / 10)
    )
    global_NW = global_coord_from_tmerc_coord(
        TmercCoord(tmerc_bounding_box_west, tmerc_bounding_box_north)
    )
    global_SE = global_coord_from_tmerc_coord(
        TmercCoord(tmerc_bounding_box_east, tmerc_bounding_box_south)
    )
    return global_NW, global_SE


def global_coord_from_tmerc_coord(tmerc_coord: TmercCoord) -> GlobalCoord:
    """

    :param x: x coordinate (easting) - It's axis points to the right
    :param y: y cooridnate (northing) - It's axis points upwards
    :return:
    """
    tmerc_projection = get_projection()
    lon, lat = tmerc_projection(tmerc_coord.east, tmerc_coord.north, inverse=True)
    return GlobalCoord(lon, lat)


class CityModel:
    """Serves as simple data structure for a city_model. In the current
    implementation a city model is a dictionary
    containing buildings. The key is the building's id and the value is another
    dictionary containing the nodes of the
    buildings outer contour line (polygon) and the buildings height. A buildings node
    is described by its longitudinal
    and latitudinal position.
    Besides the buildings dictionary there are the dimensions of the model in
     x and y direction."""

    def __init__(self):
        self.name: str | None = None
        self.scenario_info: ScenarioInfo | None = None
        self.buildings: dict | None = None
        self.semantics: dict | None = None
        self.streets: dict | None = None


def separate_relation(node_coords, ways, entity, index, dict_key, dict_val):
    """Gets a list of nodes and a list of ways the relation can consist of, the
    relation as entity, a list of used member indices, a key name and a value name
    and value name and separates the relation into a dictionary containig separate
    closed loop ways and the "main" way."""
    d = {}
    counter = 0
    connected_nodes = []
    for idx in index:
        # get the coordinates of the way
        nodes = list([item for item in ways.get(entity.members[idx].member_id, "")])
        if len(nodes) == 0:
            continue
        # check if way is closed loop
        if nodes[0] == nodes[-1]:  # if closed loop, add separate polygon
            d["".join([str(entity.id), f"_{counter}"])] = {
                "nodes": tuple(nodes),
                dict_key: dict_val,
            }
            counter += 1
        else:  # add to main polygon
            if len(connected_nodes) == 0:
                connected_nodes.extend(nodes)
            else:
                if euclidian_distance(
                    node_coords[connected_nodes[-1]], node_coords[nodes[0]]
                ) < euclidian_distance(
                    node_coords[connected_nodes[0]], node_coords[nodes[-1]]
                ):
                    connected_nodes = connected_nodes + nodes  # append at back
                else:
                    connected_nodes = nodes + connected_nodes  # append in beginning
    if len(connected_nodes) > 0:
        d[entity.id] = {"nodes": tuple(connected_nodes), dict_key: dict_val}
    return d


def str2float(numeric_string):
    string_numeric = False
    literals = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for literal in literals:
        if literal in numeric_string:
            string_numeric = True
            break
    if not string_numeric:
        raise NoNumericLiteralException("str2float", "There is no " "number in string.")
    try:
        return float(numeric_string)
    except ValueError:
        if (
            len(numeric_string.split(";")) >= 3
        ):  # sometimes there are multiple building levels in ond building chained
            return float(
                sum([float(el) for el in numeric_string.split(";")])
                / len(numeric_string.split(";"))
            )
        elif (
            len(numeric_string.split(",")) >= 3
        ):  # sometimes there are multiple building levels in ond building chained
            return float(
                sum([float(el) for el in numeric_string.split(",")])
                / len(numeric_string.split(","))
            )
        elif " m" in numeric_string:
            return float(numeric_string.strip(" ")[0])
        elif "m" in numeric_string:
            return float(numeric_string.strip("m")[0])
        elif ", " in numeric_string:
            return float(numeric_string.replace(", ", "."))
        elif "," in numeric_string:
            return float(numeric_string.replace(",", "."))
        elif ";" in numeric_string:
            return float(numeric_string.replace(";", "."))
        elif "'" in numeric_string:
            return float(numeric_string.strip("'")[0]) * 0.3048  # foot to meter
        else:
            assert (
                1 == 0
            ), f"Conversion not defined with following value: {numeric_string}"


def str2int(numeric_string):
    string_numeric = False
    literals = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for literal in literals:
        if literal in numeric_string:
            string_numeric = True
            break
    if not string_numeric:
        raise NoNumericLiteralException("str2float", "There is" " no number in string.")
    try:
        return int(numeric_string)
    except ValueError:
        if (
            ";" in numeric_string and len(numeric_string.split(";")) >= 2
        ):  # sometimes there are multiple building levels in ond building chained
            return int(
                sum([float(el) for el in numeric_string.split(";")])
                / len(numeric_string.split(";"))
            )
        else:
            assert (
                1 == 0
            ), f"Conversion not defined with following value: {numeric_string}"


def get_bounding_box(scenario_info: ScenarioInfo) -> tuple[float, float, float, float]:
    global_NW, global_SE = get_global_osm_bounding_box(scenario_info)
    top = global_NW.lat
    left = global_NW.lon
    bottom = global_SE.lat
    right = global_SE.lon
    return top, left, bottom, right


class OsmExporter:
    """This class is responsible for calculating a city model from data of an
    openstreetmap export."""

    def __init__(self, path_to_osm_file: str, data_save_folder):
        self.path_to_osm_file = path_to_osm_file
        self.data_save_folder = data_save_folder

    def extract_osm_data(self, scenario_info: ScenarioInfo):
        if not os.path.exists(os.path.join(self.data_save_folder, "osm")):
            os.makedirs(os.path.join(self.data_save_folder, "osm"))

        # extract geospatial bounding box data from *.osm.bz2 file
        self.extract_geospatial_data_from_osm(scenario_info)
        # extract building information from bounding box file
        self.extract_building_information(scenario_info)
        # extract street information from bounding box file
        self.extract_street_information(scenario_info)
        # extract semantic information from bounding box file
        self.extract_semantic_information(scenario_info)

    def calculate_city_model(self, scenario_info: ScenarioInfo):
        self.extract_osm_data(scenario_info)
        if not os.path.exists(os.path.join(self.data_save_folder, "city_models")):
            os.makedirs(os.path.join(self.data_save_folder, "city_models"))

        city_model = CityModel()
        city_model.name = scenario_info.user_identifier
        city_model.scenario_info = scenario_info
        city_model.buildings = self.process_building_information(scenario_info)
        city_model.semantics = self.process_semantic_information(scenario_info)
        city_model.streets = self.process_street_information(scenario_info)

        return city_model

    def extract_building_information(self, scenario_info: ScenarioInfo):
        osm_bounding_box_file_path = self.get_osm_bounding_box_path(scenario_info)
        osm_buildings_file_path = self.get_osm_buildings_path(scenario_info)
        if os.path.exists(osm_buildings_file_path):
            print(
                f"Geospatial OSM building file {osm_buildings_file_path} "
                f"already exists."
            )
        else:
            print(
                "Extracting buildings from bounding box area. This my take a while ..."
            )
            # extract the building information from area in the bounding, if
            # it has not already been done
            command = (
                f"osmosis --read-xml file={osm_bounding_box_file_path} --log-progress "
                f"--tf reject-relations --tf accept-ways building=* building:part=* "
                f"--used-node --read-xml file={osm_bounding_box_file_path} "
                f"--log-progress --tf accept-relations type=associated_entrance "
                f"building=* building:part=* --used-way --used-node"
                f" --merge --write-xml {osm_buildings_file_path}"
            )
            print(command)
            subprocess.run(command, shell=True)

    def get_osm_buildings_path(self, scenario_info):
        osm_buildings_file_path = os.path.join(
            self.data_save_folder,
            "osm",
            f"{get_osm_identifier(scenario_info)}_buildings.osm",
        )
        return osm_buildings_file_path

    def extract_street_information(self, scenario_info: ScenarioInfo):
        osm_bounding_box_file_path = self.get_osm_bounding_box_path(scenario_info)
        osm_streets_file_path = self.get_osm_streets_path(scenario_info)
        if os.path.exists(osm_streets_file_path):
            print(f"Geospatial OSM street file {osm_streets_file_path} already exists.")
        else:
            print("Extracting streets from bounding box area. This my take a while ...")
            # extract the building information from area in the bounding, if it
            # has not already been done
            command = (
                f"osmosis --read-xml file={osm_bounding_box_file_path} --log-progress "
                f"--tf reject-relations --tf accept-ways highway=* --used-node "
                f"--write-xml {osm_streets_file_path}"
            )
            print(command)
            subprocess.run(command, shell=True)

    def get_osm_streets_path(self, scenario_info):
        osm_streets_file_path = os.path.join(
            self.data_save_folder,
            "osm",
            f"{get_osm_identifier(scenario_info)}_streets.osm",
        )
        return osm_streets_file_path

    def extract_semantic_information(self, scenario_info: ScenarioInfo):
        osm_bounding_box_file_path = self.get_osm_bounding_box_path(scenario_info)
        osm_semantics_file_path = self.get_osm_semantics_path(scenario_info)
        if os.path.exists(osm_semantics_file_path):
            print(
                f"Geospatial OSM semantic file {osm_semantics_file_path}"
                f" already exists."
            )
        else:
            print(
                "Extracting semantic information from bounding box area. "
                "This my take a while ..."
            )
            # extract the building information from area in the bounding,
            # if it has not already been done
            command = (
                f"osmosis --read-xml file={osm_bounding_box_file_path} --log-progress "
                f"--tf reject-relations --tf accept-ways amenity=* leisure=* natural=* "
                f"--used-node --read-xml file={osm_bounding_box_file_path} "
                f"--log-progress --tf reject-relations --tf accept-ways landuse=* "
                f"area=yes --used-node --read-xml file={osm_bounding_box_file_path}"
                f" --log-progress --tf accept-relations leisure=* natural=* landuse=* "
                f"amenity=* --used-way --used-node --merge "
                f"--merge --write-xml {osm_semantics_file_path}"
            )
            print(command)
            subprocess.run(command, shell=True)

    def get_osm_semantics_path(self, scenario_info):
        osm_semantics_file_path = os.path.join(
            self.data_save_folder,
            "osm",
            f"{get_osm_identifier(scenario_info)}_semantics.osm",
        )
        return osm_semantics_file_path

    def extract_geospatial_data_from_osm(self, scenario_info: ScenarioInfo):
        """It is assumed that in the path <path_to_osm_file> there is saved a
        osm export file (e.g. downloaded from
         http://download.geofabrik.de/) as *.osm.bz2.
        """
        osm_bounding_box_file_path = self.get_osm_bounding_box_path(scenario_info)

        if os.path.exists(osm_bounding_box_file_path):
            print(
                f"Geospatial OSM bounding box export {osm_bounding_box_file_path} "
                f"already exists."
            )
        else:
            top, left, bottom, right = get_bounding_box(scenario_info)
            print(
                "Extracting bounding box area from *.osm file. This may take a while."
            )
            # extract the area in the bounding box from the osm export file, if it has
            # not already been done
            command = (
                f"osmosis --read-xml enableDateParsing=no file={self.path_to_osm_file} "
                f"--bounding-box top={str(top)} left={str(left)} bottom={str(bottom)} "
                f"right={str(right)} completeWays=yes completeRelations=yes "
                f"clipIncompleteEntities=true --write-xml {osm_bounding_box_file_path}"
            )
            print(command)
            subprocess.run(command, shell=True)

    def get_osm_bounding_box_path(self, scenario_info):
        geospatial_data_name = f"{get_osm_identifier(scenario_info)}.osm"
        path_to_geospatial_data = os.path.join(
            self.data_save_folder, "osm", geospatial_data_name
        )
        return path_to_geospatial_data

    def process_building_information(self, scenario_info: ScenarioInfo) -> dict:
        nodes = {}
        buildings = {}
        ways = {}
        way_no_building = 0
        relation_no_building = 0

        for entity in parse_file(self.get_osm_buildings_path(scenario_info)):
            if isinstance(entity, Node):  # process OSM nodes
                nodes[entity.id] = (entity.lon, entity.lat)
            elif isinstance(entity, Way):  # process OSM ways
                ways[entity.id] = entity.nodes
                if "building" in entity.tags:
                    # avoid underground buildings
                    if "layer" in entity.tags:
                        layer = str2int(entity.tags["layer"])
                        if layer < 0:
                            continue
                    try:
                        if "height" in entity.tags:
                            buildings[entity.id] = {
                                "nodes": entity.nodes,
                                "height": str2float(entity.tags["height"]),
                            }
                        elif "building:levels" in entity.tags:
                            buildings[entity.id] = {
                                "nodes": entity.nodes,
                                "height": str2float(entity.tags["building:levels"]) * 3,
                            }
                        else:
                            buildings[entity.id] = {"nodes": entity.nodes, "height": 10}
                    except NoNumericLiteralException:
                        buildings[entity.id] = {"nodes": entity.nodes, "height": 10}
                else:
                    way_no_building += 1
            elif isinstance(entity, Relation):  # process OSM relations
                if "building" in entity.tags:
                    # avoid underground buildings
                    if "layer" in entity.tags:
                        layer = int(entity.tags["layer"])
                        if layer < 0:
                            continue
                    if "multipolygon" in entity.tags.values():
                        index = [
                            i
                            for i in range(0, len(entity.members))
                            if entity.members[i].role == "outer"
                        ]
                        if "height" in entity.tags:
                            dict_val = str2float(entity.tags["height"])
                            buildings.update(
                                separate_relation(
                                    nodes, ways, entity, index, "height", dict_val
                                )
                            )
                        elif "building:levels" in entity.tags:
                            dict_val = str2float(entity.tags["building:levels"]) * 3
                            buildings.update(
                                separate_relation(
                                    nodes, ways, entity, index, "height", dict_val
                                )
                            )
                        else:
                            dict_val = 10
                            buildings.update(
                                separate_relation(
                                    nodes, ways, entity, index, "height", dict_val
                                )
                            )
                else:
                    relation_no_building += 1
        buildings = self.convert_coordinates(buildings, nodes)
        return buildings

    def convert_coordinates(self, areas, nodes):
        # convert global coordiantes (longitude and latitude) to
        # tmerc coordinates (east north)
        for area in areas.values():
            positions = []
            for node in area["nodes"]:
                try:
                    tmerc_coord = tmerc_coord_from_global_coord(
                        GlobalCoord(nodes[node][0], nodes[node][1])
                    )
                except KeyError:
                    continue
                positions.append((tmerc_coord.east, tmerc_coord.north))
            area["pos"] = positions  # order_points(positions, 0)
            del area["nodes"]
        return areas

    def process_semantic_information(self, scenario_info: ScenarioInfo) -> dict:
        input_tags = {
            6.4: ["coastline"],
            19.2: [],
            32: [
                "retail",
                "pharmacy",
                "library",
                "arts_centre",
                "social_centre",
                "sports_centre",
                "fitness_centre",
                "theatre",
                "stadium",
                "toilets",
                "restaurant",
                "ice_rink",
                "water_park",
                "bank",
                "bus_station",
                "events_centre",
                "cafe",
                "concert_hall",
                "studio",
                "pub",
                "childcare",
                "fast_food",
                "nightclub",
                "bar",
                "brokerarge",
                "cinema",
                "bowling_alley",
                "clinic",
                "doctors",
                "post_office",
                "swingerclub",
                "dentist",
                "marketplace",
                "bicycle_rental",
                "ferry_terminal",
            ],
            44.8: [
                "commercial",
                "fuel",
                "recycling",
                "boat_rental",
                "car_rental",
                "storage_rental",
                "office",
                "car_wash",
                "cruise_terminal",
                "plant_nursery",
                "slipway",
                "highway",
            ],
            57.6: ["forest", "wood", "wetland", "earth_bank"],
            70.4: [
                "grass",
                "meadow",
                "farmland",
                "recreation_ground",
                "sand",
                "scrub",
                "common",
                "cliff",
                "bare_rock",
                "grassland",
                "mountain_range",
                "greenhouse_horticulture",
                "shingle",
                "nature_reserve",
                "land",
                "stable",
                "heath",
                "scree",
            ],
            83.2: ["school", "kindergarten", "university", "college", "playground"],
            96: [
                "hospital",
                "community_centre",
                "public_building",
                "conference_centre",
                "courthouse",
                "education",
                "prep_school",
                "schoolyard",
            ],
            108.8: ["residential"],
            121.6: ["reserved: general_street_information"],
            134.4: [
                "parking",
                "fairground",
                "garages",
                "parking_space",
                "parklet",
                "bicycle_parking",
                "motorcycle_parking",
            ],
            147.2: [],
            160: ["other"],
            172.8: ["reserved: general_building_information"],
            185.6: ["industrial", "brownfield", "construction", "wasteland", "depot"],
            198.4: [
                "military",
                "railway",
                "fire_station",
                "ranger_station",
                "police",
                "prison",
            ],
            211.2: [
                "allotments",
                "village_green",
                "park",
                "cemetery",
                "shelter",
                "fitness_station",
                "tree_row",
                "practice_pitch",
                "farmyard",
                "golf_course",
                "pitch",
                "garden",
                "bleachers",
                "dog_park",
                "track",
                "beach",
                "bench",
                "disc_golf_course",
                "biergarten",
                "bleacher",
                "outdoor_seating",
                "flowerbed",
            ],
            224: [
                "place_of_worship",
                "religious",
                "fraternity",
                "social_facility",
                "churchyard",
            ],
            236.8: ["basin", "swimming_pool", "fountain", "reservoir"],
            249.6: ["water", "ocean", "marina", "reef", "strait"],
        }  # 'bay'-> destroys costline while rendering

        # invert input dictionary
        semantic_values = {}
        for key, val in input_tags.items():
            for entry in val:
                semantic_values[entry] = key

        nodes: dict[int, tuple[float, float]] = {}  # unconverted_node_information
        semantic_information: dict[int, dict] = {}
        ways: dict[int, tuple] = {}
        not_used: dict[str, int] = {}

        for entity in parse_file(self.get_osm_semantics_path(scenario_info)):
            if isinstance(entity, Node):  # process OSM nodes
                nodes[entity.id] = (entity.lon, entity.lat)
            elif isinstance(entity, Way):  # process OSM ways
                ways[entity.id] = entity.nodes
                if "natural" in entity.tags:
                    try:
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values[entity.tags["natural"]],
                        }
                    except KeyError:
                        print(
                            "No special_pos_color tag for natural {}".format(
                                entity.tags["natural"]
                            )
                        )
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values["other"],
                        }
                        if entity.tags["natural"] in not_used:
                            not_used[entity.tags["natural"]] += 1
                        else:
                            not_used[entity.tags["natural"]] = 1
                elif "landuse" in entity.tags:
                    try:
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values[entity.tags["landuse"]],
                        }
                    except KeyError:
                        print(
                            "No special_pos_color tag for landuse {}".format(
                                entity.tags["landuse"]
                            )
                        )
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values["other"],
                        }
                        if entity.tags["landuse"] in not_used:
                            not_used[entity.tags["landuse"]] += 1
                        else:
                            not_used[entity.tags["landuse"]] = 1
                elif "leisure" in entity.tags:
                    try:
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values[entity.tags["leisure"]],
                        }
                    except KeyError:
                        print(
                            "No special_pos_color tag for leisure {}".format(
                                entity.tags["leisure"]
                            )
                        )
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values["other"],
                        }
                        if entity.tags["leisure"] in not_used:
                            not_used[entity.tags["leisure"]] += 1
                        else:
                            not_used[entity.tags["leisure"]] = 1
                elif "amenity" in entity.tags:
                    try:
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values[entity.tags["amenity"]],
                        }
                    except KeyError:
                        print(
                            "No special_pos_color tag for amenity {}".format(
                                entity.tags["amenity"]
                            )
                        )
                        semantic_information[entity.id] = {
                            "nodes": entity.nodes,
                            "color_tag": semantic_values["other"],
                        }
                        if entity.tags["amenity"] in not_used:
                            not_used[entity.tags["amenity"]] += 1
                        else:
                            not_used[entity.tags["amenity"]] = 1
            elif isinstance(entity, Relation):  # process OSM relations
                if "multipolygon" in entity.tags.values():
                    index = [
                        i
                        for i in range(0, len(entity.members))
                        if entity.members[i].role in ["outer", ""]
                    ]
                    if "natural" in entity.tags:
                        try:
                            dict_val = semantic_values[entity.tags["natural"]]
                            semantic_information.update(
                                separate_relation(
                                    nodes, ways, entity, index, "color_tag", dict_val
                                )
                            )
                        except KeyError:
                            print(
                                f'No special_pos_color tag for relation natural '
                                f'{entity.tags["natural"]}'
                            )
                    elif "landuse" in entity.tags:
                        try:
                            dict_val = semantic_values[entity.tags["landuse"]]
                            semantic_information.update(
                                separate_relation(
                                    nodes, ways, entity, index, "color_tag", dict_val
                                )
                            )
                        except KeyError:
                            print(
                                f'No special_pos_color tag for relation landuse '
                                f'{entity.tags["landuse"]}'
                            )
                    elif "leisure" in entity.tags:
                        try:
                            dict_val = semantic_values[entity.tags["leisure"]]
                            semantic_information.update(
                                separate_relation(
                                    nodes, ways, entity, index, "color_tag", dict_val
                                )
                            )
                        except KeyError:
                            print(
                                f'No special_pos_color tag for relation leisure '
                                f'{entity.tags["leisure"]}'
                            )
                    elif "amenity" in entity.tags:
                        try:
                            dict_val = semantic_values[entity.tags["amenity"]]
                            semantic_information.update(
                                separate_relation(
                                    nodes, ways, entity, index, "color_tag", dict_val
                                )
                            )
                        except KeyError:
                            print(
                                f'No special_pos_color tag for relation amenity '
                                f'{entity.tags["amenity"]}'
                            )
        semantic_information = self.convert_coordinates(semantic_information, nodes)
        if len(not_used) > 0:
            print(
                "There are undefined tags. To include them in the map extraction "
                "process:"
                "\n1) add them to the process_semantic_information() function "
                "in osm_exporter.py,"
                "\n2) delete the respective *_semantics.osm file in the osm folder"
                "\n3) delete the *_city_model.pkl file in the city_model folder, and"
                "\n4) re-run met.py"
            )
            print("Not defined tags are:")
            print(not_used)
        return semantic_information

    def process_street_information(self, scenario_info: ScenarioInfo) -> dict:
        nodes = {}
        streets = {}
        ways = {}

        for entity in parse_file(self.get_osm_streets_path(scenario_info)):
            if isinstance(entity, Node):  # process OSM nodes
                nodes[entity.id] = (entity.lon, entity.lat)
            elif isinstance(entity, Way):  # process OSM ways
                ways[entity.id] = entity.nodes
                if "highway" in entity.tags:
                    streets[entity.id] = {"nodes": entity.nodes, "color_tag": 55}

        streets = self.convert_coordinates(streets, nodes)
        return streets
