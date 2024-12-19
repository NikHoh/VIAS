import os

from vias import mopp


def main():
    path_start_lon = 2.291776
    path_start_lat = 48.856548
    path_goal_lon = 2.297130
    path_goal_lat = 48.860042

    base_data_folder = r"../../VIAS_data/"
    data_input_folder = os.path.join(base_data_folder, "input")
    data_processing_folder = os.path.join(base_data_folder, "processing")
    data_save_folder = os.path.join(base_data_folder, "output")
    path_to_config = r"../src/vias/mopp_config.yaml"

    mopp.main(
        base_data_folder,
        path_start_lon,
        path_start_lat,
        path_goal_lon,
        path_goal_lat,
        path_to_config,
        data_input_folder,
        data_save_folder,
        data_processing_folder,
    )


if __name__ == "__main__":
    main()
