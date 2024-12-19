from vias import mopp
from vias.utils.helpers import ScenarioInfo, GlobalCoord


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

    path_start = GlobalCoord(2.291776, 48.856548)
    path_goal = GlobalCoord(2.297130, 48.860042)

    data_input_folder = r"<insert_path>/mopp_test/input/"
    data_save_folder = r"<insert_path>/mopp_test/output/"
    path_to_config = r"../mopp_config.yaml"
    data_processing_folder = r"<insert_path>/mopp_test/processing/"

    mopp.main(scenario_info, path_start, path_goal, path_to_config, data_input_folder, data_save_folder,
              data_processing_folder)


if __name__ == '__main__':
    main()
