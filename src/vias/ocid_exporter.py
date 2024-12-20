import copy as cp
import math
import os

import numpy as np
import pandas as pd

# import pickle5 as pkl
from .osm_exporter import get_global_osm_bounding_box
from .scenario import tmerc_coord_from_global_coord
from .utils.helpers import (
    GlobalCoord,
    ScenarioInfo,
    TmercCoord,
    get_osm_identifier,
)


def get_tmerc_ocid_bounding_box(scenario_info) -> tuple[TmercCoord, TmercCoord]:
    global_NW, global_SE = get_global_osm_bounding_box(scenario_info)
    return tmerc_coord_from_global_coord(global_NW), tmerc_coord_from_global_coord(
        global_SE
    )


class OcidExporter:
    """This class is responsible for extracting Cell Tower positions from a CVS file
    downloaded from
    https://opencellid.org/#zoom=16&lat=37.77889&lon=-122.41942"""

    def __init__(self, path_to_ocid_files: str, data_save_folder):
        print("Init OCID Reader")
        self.path_to_ocid_files = path_to_ocid_files
        self.data_save_folder = data_save_folder

    def get_ocid_bounding_box_path(self, scenario_info):
        geospatial_data_name = f"{get_osm_identifier(scenario_info)}.csv"
        path_to_geospatial_data = os.path.join(
            self.data_save_folder, "ocid", geospatial_data_name
        )
        return path_to_geospatial_data

    def extract_geospatial_data_from_ocid(self, scenario_info: ScenarioInfo):
        """It is assumed that in the path <path_to_ocid_file> there is saved a csv.gz
        export file (e.g. downloaded from
        https://opencellid.org/
        """
        if not os.path.exists(os.path.join(self.data_save_folder, "ocid")):
            os.makedirs(os.path.join(self.data_save_folder, "ocid"))
        ocid_bounding_box_file_path = self.get_ocid_bounding_box_path(scenario_info)
        ocid_bounding_box_file: None | pd.DataFrame = None
        if os.path.exists(ocid_bounding_box_file_path):
            print(
                f"Geospatial OCID bounding box export {ocid_bounding_box_file_path} "
                f"already exists."
            )
        else:
            tmerc_NW, tmerc_SE = get_tmerc_ocid_bounding_box(scenario_info)
            left = tmerc_NW.east  # do not delete, is used below in panda's query string
            top = tmerc_NW.north
            right = tmerc_SE.east
            bottom = tmerc_SE.north
            print(
                "Extracting bounding box area from ocid file. This may take a while ..."
            )
            # extract the area in the bounding box from the ocid export file,
            # if it has not already been done
            column_names = [
                "radio_type",
                "MCC",
                "MNC",
                "LAC",
                "CID",
                "unit",
                "long",
                "lat",
                "range",
                "samples",
                "changeable",
                "created",
                "updated",
                "averageSignal",
            ]
            for _, _, files in os.walk(self.path_to_ocid_files):
                for name in files:
                    if "filtered_data" in name:
                        continue
                    with pd.read_csv(
                        os.path.join(self.path_to_ocid_files, name),
                        header=None,
                        chunksize=10**6,
                    ) as reader:
                        for df in reader:
                            df.rename(
                                columns=dict(
                                    zip(
                                        range(0, len(column_names)),
                                        column_names,
                                        strict=False,
                                    )
                                ),
                                inplace=True,
                            )

                            tmerc_coords = [
                                tmerc_coord_from_global_coord(GlobalCoord(lon, lat))
                                for lat, lon in zip(
                                    *[np.array(df.lat), np.array(df.long)], strict=False
                                )
                            ]

                            east = [coord.east for coord in tmerc_coords]
                            north = [coord.north for coord in tmerc_coords]
                            df.insert(len(df.columns), "east", east)
                            df.insert(len(df.columns), "north", north)
                            query_msg = (
                                "east >= @left and north >= @bottom and east "
                                "<= @right and north <= @top"
                            )
                            df.query(
                                query_msg,
                                local_dict={
                                    "left": left,
                                    "top": top,
                                    "right": right,
                                    "bottom": bottom,
                                },
                                inplace=True,
                            )  # filter to bounding box
                            if isinstance(ocid_bounding_box_file, type(None)):
                                ocid_bounding_box_file = cp.deepcopy(df)
                            else:
                                ocid_bounding_box_file = pd.merge(
                                    ocid_bounding_box_file, cp.deepcopy(df), how="outer"
                                )
            print("Save filtered data.")
            assert ocid_bounding_box_file is not None
            ocid_bounding_box_file.to_csv(ocid_bounding_box_file_path)
            print("Saving done.")

    def get_cell_positions(
        self, scenario_info: ScenarioInfo, radio_type="4G", min_samples=50
    ):
        self.extract_geospatial_data_from_ocid(scenario_info)

        ocid_bounding_box_file_path = self.get_ocid_bounding_box_path(scenario_info)
        ocid_bounding_box_file: pd.DataFrame = pd.read_csv(
            ocid_bounding_box_file_path, index_col=False
        )
        unique_types = ocid_bounding_box_file.radio_type.unique()
        for rtype in unique_types:
            if rtype not in ["NR", "LTE", "UMTS", "CDMA", "GSM"]:
                assert (
                    1 == 0
                ), f"Add type {rtype} to expression: It has never been needed so far."
        if radio_type == "5G":
            if "NR" in unique_types:
                expression = "radio_type == 'NR'"
            else:
                return []
        elif radio_type == "4G":
            if "LTE" in unique_types:
                expression = "radio_type == 'LTE'"
            else:
                return []
        elif radio_type == "3G":
            if "UMTS" in unique_types and "CDMA" in unique_types:
                expression = "radio_type == 'UMTS' or radio_type == 'CDMA'"
            elif "UMTS" in unique_types:
                expression = "radio_type == 'UMTS'"
            elif "CDMA" in unique_types:
                expression = "radio_type == 'CDMA'"
            else:
                return []
        elif radio_type == "2G":
            if "GSM" in unique_types:
                expression = "radio_type == 'GSM'"
            else:
                return []
        else:
            assert 1 == 0, "Radio type not defined"
        if min_samples < math.inf:
            expression += f" and samples >= {min_samples}"
        print(f"Filter {radio_type} data from radio types")
        queried_data = ocid_bounding_box_file.query(expression, inplace=False)
        sliced_data = queried_data.loc[:, "east":"north"]  # type: ignore[misc]
        return_data = np.array(sliced_data).tolist()
        print("Filtering done.")
        return return_data
