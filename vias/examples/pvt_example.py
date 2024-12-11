from vias import pvt

def main():
    path_to_grid_maps_folder = r"<insert_path>/mopp_test/input/"
    path_to_optimization_results = r"<insert_path>/mopp_test/output/MOPP_paris_test_<...>/"
    path_to_config = r"../pvt_config.yaml"

    pvt.main(path_to_config, path_to_optimization_results, path_to_grid_maps_folder=path_to_grid_maps_folder)


if __name__ == '__main__':
    main()
