import unittest
from dataclasses import astuple

import numpy as np

from vias.grid_map import GridMap
from vias.utils.helpers import ArrayCoord, LocalCoord, MapInfo


def get_test_map():
    return GridMap(MapInfo(2.5, 40.0, 0.0, 0.0, 20, 10, 10, 4, 2, 5, "test_map"))


class TestGridMap(unittest.TestCase):
    def test_local_coord_meshgrid(self):
        grid_map = get_test_map()
        xv, yv, zv = astuple(grid_map.local_coord_meshgrid)
        self.assertEqual(xv[0, 0, 0], 0.0)
        self.assertEqual(yv[0, 0, 0], 8.0)
        self.assertEqual(zv[0, 0, 0], 0.0)
        self.assertEqual(xv[1, 0, 0], 0.0)
        self.assertEqual(yv[1, 0, 0], 6.0)
        self.assertEqual(zv[1, 0, 0], 0.0)
        self.assertEqual(xv[2, 0, 0], 0.0)
        self.assertEqual(yv[2, 0, 0], 4.0)
        self.assertEqual(zv[2, 0, 0], 0.0)
        self.assertEqual(xv[3, 0, 0], 0.0)
        self.assertEqual(yv[3, 0, 0], 2.0)
        self.assertEqual(zv[3, 0, 0], 0.0)
        self.assertEqual(xv[4, 0, 0], 0.0)
        self.assertEqual(yv[4, 0, 0], 0.0)
        self.assertEqual(zv[4, 0, 0], 0.0)
        self.assertEqual(xv[0, 1, 0], 4.0)
        self.assertEqual(yv[0, 1, 0], 8.0)
        self.assertEqual(zv[0, 1, 0], 0.0)
        self.assertEqual(xv[1, 1, 0], 4.0)
        self.assertEqual(yv[1, 1, 0], 6.0)
        self.assertEqual(zv[1, 2, 0], 0.0)
        self.assertEqual(xv[2, 1, 0], 4.0)
        self.assertEqual(yv[2, 1, 0], 4.0)
        self.assertEqual(zv[2, 1, 0], 0.0)
        self.assertEqual(xv[3, 1, 0], 4.0)
        self.assertEqual(yv[3, 1, 0], 2.0)
        self.assertEqual(zv[3, 1, 0], 0.0)
        self.assertEqual(xv[4, 1, 0], 4.0)
        self.assertEqual(yv[4, 1, 0], 0.0)
        self.assertEqual(zv[4, 1, 0], 0.0)
        self.assertEqual(zv[0, 0, 1], 5.0)
        self.assertEqual(zv[1, 0, 1], 5.0)
        self.assertEqual(zv[2, 0, 1], 5.0)
        self.assertEqual(zv[3, 0, 1], 5.0)
        self.assertEqual(zv[4, 0, 1], 5.0)
        self.assertEqual(zv[0, 1, 1], 5.0)
        self.assertEqual(zv[1, 2, 1], 5.0)
        self.assertEqual(zv[2, 1, 1], 5.0)
        self.assertEqual(zv[3, 1, 1], 5.0)
        self.assertEqual(zv[4, 1, 1], 5.0)

    def test_array_coord_meshgrid(self):
        grid_map = get_test_map()
        row_v, col_v, lay_v = astuple(grid_map.array_coord_meshgrid)
        self.assertEqual(row_v[0, 0, 0], 0)
        self.assertEqual(col_v[0, 0, 0], 0)
        self.assertEqual(lay_v[0, 0, 0], 0)
        self.assertEqual(row_v[1, 0, 0], 1)
        self.assertEqual(col_v[1, 0, 0], 0)
        self.assertEqual(lay_v[1, 0, 0], 0)
        self.assertEqual(row_v[2, 0, 0], 2)
        self.assertEqual(col_v[2, 0, 0], 0)
        self.assertEqual(lay_v[2, 0, 0], 0)
        self.assertEqual(row_v[3, 0, 0], 3)
        self.assertEqual(col_v[3, 0, 0], 0)
        self.assertEqual(lay_v[3, 0, 0], 0)
        self.assertEqual(row_v[4, 0, 0], 4)
        self.assertEqual(col_v[4, 0, 0], 0)
        self.assertEqual(lay_v[4, 0, 0], 0)
        self.assertEqual(row_v[0, 1, 0], 0)
        self.assertEqual(col_v[0, 1, 0], 1)
        self.assertEqual(lay_v[0, 1, 0], 0)
        self.assertEqual(row_v[1, 1, 0], 1)
        self.assertEqual(col_v[1, 1, 0], 1)
        self.assertEqual(lay_v[1, 1, 0], 0)
        self.assertEqual(row_v[2, 1, 0], 2)
        self.assertEqual(col_v[2, 1, 0], 1)
        self.assertEqual(lay_v[2, 1, 0], 0)
        self.assertEqual(row_v[3, 1, 0], 3)
        self.assertEqual(col_v[3, 1, 0], 1)
        self.assertEqual(lay_v[3, 1, 0], 0)
        self.assertEqual(row_v[4, 1, 0], 4)
        self.assertEqual(col_v[4, 1, 0], 1)
        self.assertEqual(lay_v[4, 1, 0], 0)
        self.assertEqual(lay_v[0, 0, 1], 1)
        self.assertEqual(lay_v[1, 0, 1], 1)
        self.assertEqual(lay_v[2, 0, 1], 1)
        self.assertEqual(lay_v[3, 0, 1], 1)
        self.assertEqual(lay_v[4, 0, 1], 1)
        self.assertEqual(lay_v[0, 1, 1], 1)
        self.assertEqual(lay_v[1, 1, 1], 1)
        self.assertEqual(lay_v[1, 1, 1], 1)
        self.assertEqual(lay_v[2, 1, 1], 1)
        self.assertEqual(lay_v[3, 1, 1], 1)
        self.assertEqual(lay_v[4, 1, 1], 1)

    def test_local_coord_from_array_coord(self):
        """Use local_coord_meshgrid and array_coord_meshgrid (assuming this is
        implemented and tested)
        to examples this function."""
        grid_map = get_test_map()
        xv, yv, zv = astuple(grid_map.local_coord_meshgrid)
        row_v, col_v, lay_v = astuple(grid_map.array_coord_meshgrid)

        stacked = np.stack((row_v, col_v, lay_v), axis=3)
        for i, j, k in np.ndindex(stacked.shape[:3]):
            array_indices = stacked[i, j, k]
            array_cord = ArrayCoord(*array_indices)
            local_cord = grid_map.local_coord_from_array_coord(array_cord)
            self.assertEqual(local_cord.x, xv[i, j, k])
            self.assertEqual(local_cord.y, yv[i, j, k])
            self.assertEqual(local_cord.z, zv[i, j, k])

    def test_local_coords_from_array_coords(self):
        """Use array_coord_meshgrid and local_coord_from_array_coord() (assuming this
        is implemented and tested)
        to examples this function."""
        array_coords = []
        grid_map = get_test_map()
        row_v, col_v, lay_v = astuple(grid_map.array_coord_meshgrid)
        stacked = np.stack((row_v, col_v, lay_v), axis=3)
        for i, j, k in np.ndindex(stacked.shape[:3]):
            array_indices = stacked[i, j, k]
            array_coords.append(ArrayCoord(*array_indices))
        local_coords = grid_map.local_coords_from_array_coords(array_coords)
        for counter, (i, j, k) in enumerate(np.ndindex(stacked.shape[:3])):
            array_indices = stacked[i, j, k]
            array_cord = ArrayCoord(*array_indices)
            local_cord = grid_map.local_coord_from_array_coord(array_cord)
            self.assertEqual(local_cord, local_coords[counter])

    def test_array_coord_from_local_coord(self):
        """Use local_coord_meshgrid and array_coord_meshgrid (assuming this is
        implemented and tested)
        to examples this function."""
        grid_map = get_test_map()
        xv, yv, zv = astuple(grid_map.local_coord_meshgrid)
        row_v, col_v, lay_v = astuple(grid_map.array_coord_meshgrid)

        stacked = np.stack((xv, yv, zv), axis=3)
        for i, j, k in np.ndindex(stacked.shape[:3]):
            local_coords = stacked[i, j, k]
            local_coord = LocalCoord(*local_coords)
            array_coord = grid_map.array_coord_from_local_coord(local_coord)
            self.assertEqual(array_coord.row, row_v[i, j, k])
            self.assertEqual(array_coord.col, col_v[i, j, k])
            self.assertEqual(array_coord.lay, lay_v[i, j, k])

    def test_array_coords_from_local_coords(self):
        """Use local_coord_meshgrid and array_coord_from_local_coord() (assuming this
        is implemented and tested)
        to examples this function."""
        local_coords = []
        grid_map = get_test_map()
        xv, yv, zv = astuple(grid_map.local_coord_meshgrid)
        stacked = np.stack((xv, yv, zv), axis=3)
        for i, j, k in np.ndindex(stacked.shape[:3]):
            local_coord = stacked[i, j, k]
            local_coords.append(LocalCoord(*local_coord))
        array_coords = grid_map.array_coords_from_local_coords(local_coords)
        for counter, (i, j, k) in enumerate(np.ndindex(stacked.shape[:3])):
            local_coords = stacked[i, j, k]
            local_coord = LocalCoord(*local_coords)
            array_coord = grid_map.array_coord_from_local_coord(local_coord)
            self.assertEqual(array_coord, array_coords[counter])

    def test_get_interpolated_values_from_local_coords(self):
        grid_map = GridMap(MapInfo(2.5, 40.0, 10, 6, 8, 5, 3, 4, "test_map"))
        # define layer 0
        grid_map.grid_tensor[:, :, 0] = np.array([[1, 2], [4, 3]])
        # define layer 1
        grid_map.grid_tensor[:, :, 1] = np.array([[3, 4], [2, 1]])

        expected_regular_grid_values = {
            (0, 0, 0): 4,
            (5, 0, 0): 3,
            (0, 3, 0): 1,
            (5, 3, 0): 2,
            (0, 0, 4): 2,
            (5, 0, 4): 1,
            (0, 3, 4): 3,
            (5, 3, 4): 4,
        }
        xv, yv, zv = astuple(grid_map.local_coord_meshgrid)
        for _ in range(grid_map.shape[2]):
            stacked = np.stack((xv, yv, zv), axis=3)
            for i, j, k in np.ndindex(stacked.shape[:3]):
                local_coord = stacked[i, j, k]
                tensor_value = grid_map.get_interpolated_values_from_local_coords(
                    [LocalCoord(*local_coord)]
                )
                self.assertEqual(
                    expected_regular_grid_values[tuple(local_coord)], tensor_value
                )

        expected_interpolated_values = {
            (2.5, 0.0, 0.0): 3.5,  # top view lower layer
            (5.0, 1.5, 0.0): 2.5,
            (2.5, 3.0, 0.0): 1.5,
            (0.0, 1.5, 0.0): 2.5,
            (2.5, 1.5, 0.0): 2.5,
            (2.5, 0.0, 4.0): 1.5,  # top view upper layer
            (5.0, 1.5, 4.0): 2.5,
            (2.5, 3.0, 4.0): 3.5,
            (0.0, 1.5, 4.0): 2.5,
            (2.5, 1.5, 4.0): 2.5,
            # left view front layer
            (0.0, 0.0, 2.0): 3.0,
            (0.0, 3.0, 2.0): 2.0,
            (0.0, 1.5, 2.0): 2.5,
            (2.5, 1.5, 2.0): 2.5,
        }  # middle element
        for local_pos, expected_val in expected_interpolated_values.items():
            tensor_value = grid_map.get_interpolated_values_from_local_coords(
                [LocalCoord(*local_pos)]
            )
            self.assertEqual(expected_val, tensor_value)


if __name__ == "__main__":
    unittest.main()
