import os

import vias.mct as mct


def main():
    base_data_folder = r"../../VIAS_data/"
    data_save_folder = os.path.join(base_data_folder, "input")
    data_input_folder = os.path.join(base_data_folder, "input")
    path_to_config = r"../src/vias/mct_config.yaml"
    mct.main(base_data_folder, path_to_config, data_input_folder, data_save_folder)


if __name__ == "__main__":
    main()
