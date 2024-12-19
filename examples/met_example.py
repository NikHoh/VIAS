import os

import vias.met as met
from vias.utils.helpers import ScenarioInfo, save_scenario_info_to_json


def main():
    scenario_info = ScenarioInfo(
        2.291197, 48.860349, 500, 500, 300, 5, 5, 10, "paris_test", 60, 240
    )

    base_data_folder = r"../../VIAS_data/"
    path_to_osm_file = r"../../VIAS_data/input/osm/paris.osm.bz2"
    data_save_folder = os.path.join(base_data_folder, "input")
    path_to_ocid_files = os.path.join(base_data_folder, "input", "ocid")
    path_to_config = r"../src/vias/met_config.yaml"

    # save scenario_info
    save_scenario_info_to_json(scenario_info, base_data_folder)

    met.main(
        path_to_osm_file,
        base_data_folder,
        path_to_config,
        data_save_folder,
        path_to_ocid_files=path_to_ocid_files,
    )


if __name__ == "__main__":
    main()
