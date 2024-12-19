import os

from vias import pvt


def main():
    base_data_folder = r"../../VIAS_data/"
    path_to_grid_maps_folder = os.path.join(base_data_folder, "input")
    path_to_optimization_results = (
        r"../../VIAS_data/output/"
        r"MOPP_paris_test_oLon_2_291_oLat_48_860_x_500_y_500_z_300_resX_5_resY_5_"
        r"resZ_10_sLon_2_292_sLat_48_857_gLon_2_297_gLat48_860"
    )
    path_to_config = r"../src/vias/pvt_config.yaml"

    pvt.main(
        base_data_folder,
        path_to_config,
        path_to_optimization_results,
        path_to_grid_maps_folder=path_to_grid_maps_folder,
    )


if __name__ == "__main__":
    main()
