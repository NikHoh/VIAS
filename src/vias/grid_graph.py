import os
import pickle
from typing import TYPE_CHECKING, Union

import numpy as np
from igraph import Graph as IGraph
from numpy import ndarray
from pyproj import Proj

if TYPE_CHECKING:
    from vias.grid_map import GridMap

from vias.config import get_config
from vias.path import Path
from vias.scenario import get_tmerc_map_origin
from vias.utils.helpers import (
    ArrayCoord,
    LocalCoord,
    MapInfo,
    OutOfOperationSpace,
    ScenarioInfo,
    TmercCoord,
    get_graph_identifier,
    get_map_identifier,
)


class InvalidCoordinateError(Exception):
    """Raised when no valid path is found."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def get_local_to_tmerc_transform(
    map_info: MapInfo, tmerc_projection: Proj | None = None
) -> np.ndarray:
    tmerc_map_origin = get_tmerc_map_origin(map_info, tmerc_projection=tmerc_projection)
    T_trans = np.array(
        [
            [1, 0, tmerc_map_origin.east],
            [0, 1, tmerc_map_origin.north - map_info.y_length],
            [0, 0, 1],
        ]
    )
    T_tot = T_trans
    return T_tot


def get_tmerc_to_local_transform(
    map_info: MapInfo, tmerc_projection: Proj | None = None
) -> np.ndarray:
    return np.linalg.inv(
        get_local_to_tmerc_transform(map_info, tmerc_projection=tmerc_projection)
    )


def local_coord_from_tmerc_coord(
    tmerc_coord: TmercCoord,
    map_info: MapInfo,
    height_val=0.0,
    transform: ndarray | None = None,
) -> LocalCoord:
    if transform is None:
        transform = get_tmerc_to_local_transform(map_info)
    local_array = np.matmul(
        transform, np.array([[tmerc_coord.east], [tmerc_coord.north], [1]])
    )
    return LocalCoord(local_array[0].item(), local_array[1].item(), height_val)


def tmerc_coord_from_local_coord(
    local_coord: LocalCoord, map_info: MapInfo
) -> TmercCoord:
    transform = get_local_to_tmerc_transform(map_info)
    tmerc_array = np.matmul(
        transform, np.array([[local_coord.x], [local_coord.y], [1]])
    )
    return TmercCoord(tmerc_array[0].item(), tmerc_array[1].item())


class GridGraph:
    def __init__(self, map_info: MapInfo, graph: IGraph, dimension=3):
        self.dimension = dimension
        self.map_info = map_info
        self._inverse_grid_map_transform = None
        self._array_transform = None
        self._inverse_array_transform = None
        self._grid_map_transform = None
        self.graph = graph

    @property
    def directed(self) -> bool | None:
        if self.graph is not None:
            return self.graph.is_directed()
        else:
            return None

    def local_coord_from_tmerc_coord(
        self, tmerc_coord: TmercCoord, height_val=0.0
    ) -> LocalCoord:
        inverse_transform = self.inverse_grid_map_transform
        local_array = np.matmul(
            inverse_transform, np.array([[tmerc_coord.east], [tmerc_coord.north], [1]])
        )
        return LocalCoord(local_array[0].item(), local_array[1].item(), height_val)

    def tmerc_coord_from_local_coord(self, local_coord: LocalCoord) -> TmercCoord:
        transform = self.grid_map_transform
        tmerc_array = np.matmul(
            transform, np.array([[local_coord.x], [local_coord.y], [1]])
        )
        return TmercCoord(tmerc_array[0].item(), tmerc_array[1].item())

    def local_coord_from_array_coord(self, array_coord: ArrayCoord) -> LocalCoord:
        transform = self.array_transform
        local_array = np.matmul(
            transform,
            np.array([[array_coord.row], [array_coord.col], [array_coord.lay], [1]]),
        ).astype(float)
        return LocalCoord(
            local_array[0].item(), local_array[1].item(), local_array[2].item()
        )

    def local_coord_lies_in_map(self, local_coord: LocalCoord) -> bool:
        return (
            local_coord.x >= 0
            and local_coord.y >= 0
            and local_coord.z >= 0
            and local_coord.x < self.x_length
            and local_coord.y < self.y_length
            and local_coord.z < self.z_length
        )

    @property
    def inverse_array_transform(self):
        """
        Inverse array transform, see array_transform()
        """
        if self._inverse_array_transform is None:
            self._inverse_array_transform = np.linalg.inv(self.array_transform)
        return self._inverse_array_transform

    def array_coord_from_local_coord(self, local_coord: LocalCoord) -> ArrayCoord:
        if not self.local_coord_lies_in_map(local_coord):
            raise OutOfOperationSpace(
                "array_coord_from_local_coord()",
                f"Local coord {local_coord} lies outside of operation space.",
            )
        inverse_transform = self.inverse_array_transform
        array_array = np.matmul(
            inverse_transform,
            np.array([[local_coord.x], [local_coord.y], [local_coord.z], [1]]),
        )
        return ArrayCoord(
            int(np.ceil(array_array[0]).item()),
            int(np.floor(array_array[1]).item()),
            int(np.floor(array_array[2]).item()),
        )

    def array_coords_from_local_coords(
        self, local_coords: list[LocalCoord]
    ) -> list[ArrayCoord]:
        for local_coord in local_coords:
            if not self.local_coord_lies_in_map(local_coord):
                raise OutOfOperationSpace(
                    "array_coord_from_local_coord()",
                    f"Local coord {local_coord} lies outside of operation space.",
                )
        inverse_transform = self.inverse_array_transform
        row_vectors = np.vstack(
            [local_coord.as_homogeneous() for local_coord in local_coords]
        )
        res = inverse_transform @ row_vectors.T
        # apply floor and ceil operations
        res[0] = np.ceil(res[0])
        res[1] = np.floor(res[1])
        res[2] = np.floor(res[2])
        # cast to int
        res = res.astype(int)
        # split in row vectors and ignore 4th row (only 1s)
        res = np.hsplit(res[0:3, :], res.shape[1])
        return [ArrayCoord(res[0].item(), res[1].item(), res[2].item()) for res in res]

    def local_coords_from_array_coords(
        self, array_coords: list[ArrayCoord]
    ) -> list[LocalCoord]:
        transform = self.array_transform
        row_vectors = np.vstack(
            [array_coord.as_homogeneous() for array_coord in array_coords]
        )
        res = transform @ row_vectors.T
        # cast to float
        res = res.astype(float)
        # split in row vectors and ignore 4th row (only 1s)
        res = np.hsplit(res[0:4, :], res.shape[1])
        return [LocalCoord(res[0].item(), res[1].item(), res[2].item()) for res in res]

    @property
    def array_transform(self):
        """Calculates a transform
        | x |   | a  b  c  d | | row |
        | y | = | e  f  g  h | | col |
        | z |   | i  j  k  l | | lay |
        | 1 |   | 0  0  0  1 | | 1   |

        whereas x, y and z are the Local coordinates in meter with its coordinate frame
        located in the lower left;
        x pointing to the right, and y pointing upwards and z pointing out of the plain;

        row, col and lay are the respective numpy array indices with the origin in the
        upper left.

        """
        if self._array_transform is None:
            T_tot = np.array(
                [
                    [0, self.x_res, 0, 0],
                    [-self.y_res, 0, 0, self.y_length - self.y_res],
                    [0, 0, self.z_res, 0],
                    [0, 0, 0, 1],
                ]
            )

            self._array_transform = T_tot
        return self._array_transform

    @property
    def grid_map_transform(self):
        """Calculates an affine transform
        | east  |   | a  b  c | | x |
        | north | = | d  e  f | | y |
        | 1     |   | g  h  i | | 1 |

        whereas east and north are the spatial (e.g. tmerc) coordinates east
        pointing to the right and north pointing upwards;

        and x and y are the cartesian coordinates with its coordinate frame located in
        the lower left; x pointing to the right, and y pointing upwards.

        """
        if self._grid_map_transform is None:
            self._grid_map_transform = get_local_to_tmerc_transform(self.map_info)
        return self._grid_map_transform

    @property
    def inverse_grid_map_transform(self):
        """
        Inverse grid grid_map transform, see grid_map_transform()
        """
        if self._inverse_grid_map_transform is None:
            self._inverse_grid_map_transform = get_tmerc_to_local_transform(
                self.map_info
            )
        return self._inverse_grid_map_transform

    @property
    def x_length(self):
        return self.map_info.x_length

    @x_length.setter
    def x_length(self, value):
        self.map_info.x_length = value
        self._grid_map_transform = None
        self._inverse_grid_map_transform = None

    @property
    def x_res(self):
        return self.map_info.x_res

    @x_res.setter
    def x_res(self, value):
        self.map_info.x_res = value
        self._grid_map_transform = None
        self._inverse_grid_map_transform = None
        self._array_transform = None
        self._inverse_array_transform = None

    @property
    def y_length(self):
        return self.map_info.y_length

    @y_length.setter
    def y_length(self, value):
        self.map_info.y_length = value
        self._grid_map_transform = None
        self._inverse_grid_map_transform = None
        self._array_transform = None
        self._inverse_array_transform = None

    @property
    def y_res(self):
        return self.map_info.y_res

    @y_res.setter
    def y_res(self, value):
        self.map_info.y_res = value
        self._grid_map_transform = None
        self._inverse_grid_map_transform = None
        self._array_transform = None
        self._inverse_array_transform = None

    @property
    def z_length(self):
        return self.map_info.z_length

    @z_length.setter
    def z_length(self, value):
        self.map_info.z_length = value

    @property
    def z_res(self):
        return self.map_info.z_res

    @z_res.setter
    def z_res(self, value):
        self.map_info.z_res = value
        self._array_transform = None
        self._inverse_array_transform = None

    @property
    def name(self):
        return self.map_info.map_name

    @name.setter
    def name(self, name):
        self.map_info.map_name = name

    def plot_paths(
        self,
        paths: Path | list[Path],
        savepath="",
        suffix="",
        title="",
        prefix="",
        linestyle="",
        markerstyle=".",
        plot_lib="plotly",
    ):
        pass

    def plot_3D_paths(
        self,
        paths: list[Path] | Path,
        savepath="",
        suffix="",
        prefix="",
        plot_lib="plotly",
    ):
        pass


def save_grid_graph(
    grid_graph: Union[GridGraph, "GridMap"],
    data_save_folder: str,
    scenario_info: ScenarioInfo,
    grid_graph_name: str,
) -> None:
    """
    By default, the grid_map is saved in the "grid_maps" folder within
    the data_save_folder
    """
    save_path = get_grid_graph_path(data_save_folder, scenario_info, grid_graph_name)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path, "wb") as f:
        pickle.dump(grid_graph, f)


def load_grid_graph(
    data_save_folder: str, scenario_info: ScenarioInfo, grid_graph_name: str
) -> GridGraph:
    grid_map_path = get_grid_graph_path(
        data_save_folder, scenario_info, grid_graph_name
    )
    if os.path.exists(grid_map_path):
        with open(grid_map_path, "rb") as f:
            grid_map: GridGraph = pickle.load(f)
        return grid_map
    else:
        raise AssertionError(f"Cannot load from path {grid_map_path}")


def get_grid_graph_path(data_save_folder, scenario_info, grid_graph_name) -> str:
    """The standard grid_map dimension of all saved maps is 3."""
    if "graph" in grid_graph_name:
        assert (
            "map" not in grid_graph_name
        ), "The word map and graph are both in the name of the, avoid this."
        return os.path.join(
            data_save_folder,
            "grid_graphs",
            f"{get_graph_identifier(scenario_info)}_{grid_graph_name}.pkl",
        )
    elif "map" in grid_graph_name:
        assert (
            "graph" not in grid_graph_name
        ), "The word map and graph are both in the name of the, avoid this."
        config = get_config()
        map_dim = config.get(grid_graph_name).dimension
        return os.path.join(
            data_save_folder,
            "grid_maps",
            f"{get_map_identifier(scenario_info, map_dim)}_{grid_graph_name}.pkl",
        )
    else:
        raise AssertionError(
            f"Unknown name tag {grid_graph_name}, either the word map or the "
            f"word graph must be in the name."
        )
