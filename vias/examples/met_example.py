import vias.met as met
from vias.utils.helpers import ScenarioInfo


def main():
    scenario_info = ScenarioInfo(2.291197,
                                 48.860349,
                                 500,
                                 500,
                                 300,
                                 5,
                                 5,
                                 10,
                                 "paris_test",
                                 60,
                                 240)

    path_to_osm_file = r"<insert_path>/paris.osm.bz2"
    data_save_folder = r"<insert_path>/mopp_test/input/"
    path_to_config = r"../met_config.yaml"
    path_to_ocid_files = r"<insert_path>/mopp_test/input/ocid/FRA/"

    met.main(path_to_osm_file, scenario_info,
             path_to_config,
             data_save_folder,
             path_to_ocid_files=path_to_ocid_files)


if __name__ == '__main__':
    main()
