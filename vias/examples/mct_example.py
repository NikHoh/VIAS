import vias.mct as mct
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

    data_input_folder = r"<insert_path>/mopp_test/input/"
    data_save_folder = r"<insert_path>/mopp_test/input/"
    path_to_config = r"../mct_config.yaml"
    mct.main(scenario_info, path_to_config,
             data_input_folder, data_save_folder)


if __name__ == '__main__':
    main()
