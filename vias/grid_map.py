import copy as cp
import inspect
import itertools
import os
from dataclasses import dataclass, astuple
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.interpolate import RegularGridInterpolator

from vias.config import get_config
import plotly.graph_objects as go
from vias.grid_graph import GridGraph
from vias.osm_exporter import global_coord_from_tmerc_coord
from vias.path import Path
from vias.utils.helpers import LocalCoord, ArrayCoord, MapInfo, _save_and_close_plotly, _save_and_close_matplotlib
from vias.utils.tools import get_colors


def get_num_function_arguments(func):
    # Get the function signature
    sig = inspect.signature(func)
    # Extract the parameters from the signature
    params = sig.parameters
    # Return the number of parameters
    return len(params)


@dataclass
class MeshGrid:
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray


class GridMap(GridGraph):
    """Class that implements a grid grid_map with width and length and resolution that can be set. Values of tiles of the
    grid can be set separately or together. Values of tiles can be set or get by either the matrix entries of
    the numpy array or by providing x and y cartesian coordinates. The point (0,0) of the cartesian frame is fixed
    in the lower left corner of the GridMap."""

    def __init__(self, map_info: MapInfo, dimension=3, init_val=0.0):
        super().__init__(map_info, dimension)
        self.main_layer = 0

        self.init_grid_tensor(init_val)

        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    def __setstate__(self, state):
        # Restore the object's state (old state)
        self.__dict__.update(state)

        # Check if the new attribute exists, if not, set a default value
        if '_interpolator_with_extrapolation' not in self.__dict__:
            self._interpolator_with_extrapolation = None
        if '_interpolator_with_bound_error' not in self.__dict__:
            self._interpolator_with_bound_error = None

    @property
    def x_length(self):
        return super().x_length

    @x_length.setter
    def x_length(self, value):
        super().x_length = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    @property
    def x_res(self):
        return super().x_res

    @x_res.setter
    def x_res(self, value):
        super().x_res = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    @property
    def y_length(self):
        return super().y_length

    @y_length.setter
    def y_length(self, value):
        super().y_length = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    @property
    def y_res(self):
        return super().y_res

    @y_res.setter
    def y_res(self, value):
        super().y_res = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    @property
    def z_length(self):
        return super().z_length

    @z_length.setter
    def z_length(self, value):
        super().z_length = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    @property
    def z_res(self):
        return super().z_res

    @z_res.setter
    def z_res(self, value):
        super().z_res = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    def init_grid_tensor(self, init_val=0.0):
        if self.dimension == 3:
            self.grid_tensor = np.ones([int(self.y_length / self.y_res),
                                        int(self.x_length / self.x_res),
                                        int(self.z_length / self.z_res)]) * init_val
        elif self.dimension == 2:
            self.grid_tensor = np.ones([int(self.y_length / self.y_res),
                                        int(self.x_length / self.x_res),
                                        1]) * init_val
        else:
            assert False, "Unknown dimension"

    # for compatibility with old code, the following setters and getters are defined

    @property
    def grid_array(self):
        return self.grid_tensor[:, :, self.main_layer]

    @grid_array.setter
    def grid_array(self, new_grid_array):
        """Sets the grid grid_map (which is implemented as numpy.array)."""
        assert self.main_layer is not None, "main_layer must be set, if dimension is 3"
        assert self.grid_tensor[:, :, self.main_layer] == new_grid_array.shape
        self.grid_tensor[:, :, self.main_layer] = new_grid_array
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    def set_grid_array_as_layer(self, array: ndarray, layer: int):
        assert self.shape[0] == array.shape[0], "Matrix dimensions do not match"
        assert self.shape[1] == array.shape[1], "Matrix dimensions do not match"
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        self.grid_tensor[:, :, layer] = array

    def clear(self):
        self.init_grid_tensor()
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        self.main_layer = 0
        self.name = "cleared_map"

    @property
    def local_coord_meshgrid(self) -> MeshGrid:
        """Returns a meshgrid (see numpy meshgrid) consisting of three numpy arrays xv, yv, and zv.
        Using the indexing scheme from self.grid_tensor, i.e. an ArrayCoord, the output is the respective LocalCoord.
        Example: having the row_index=3, the col_index=2, and the lay_index=1, the respective local coordinate entries
        can be obtained by local_coord_meshgrid.xv[3], local_coord_meshgrid.yv[2], local_coord_meshgrid.zv[1]"""
        x = np.arange(0, self.x_length, self.x_res)
        y = np.flip(np.arange(0, self.y_length, self.y_res))
        # flipped as local coord frame y-axis points in different direction than array frame row axis
        z = np.arange(0, self.z_length, self.z_res)
        # x and y swapped as y-axis equals the row-axis (first axis) of the array frame
        yv, xv, zv = np.meshgrid(y, x, z, indexing='ij')
        mesh_grid = MeshGrid(xv, yv, zv)
        return mesh_grid

    @property
    def shape(self):
        """Returns the grid_tensor shape."""
        return self.grid_tensor.shape

    @property
    def array_coord_meshgrid(self) -> MeshGrid:
        """Returns a meshgrid (see numpy meshgrid) that corresponds to the self.grid_tensor indices."""
        row_indices = np.arange(0, self.shape[0], 1)
        column_indices = np.arange(0, self.shape[1], 1)
        layer_indices = np.arange(0, self.shape[2], 1)
        xv, yv, zv = np.meshgrid(row_indices, column_indices, layer_indices, indexing='ij')
        mesh_grid = MeshGrid(xv, yv, zv)
        return mesh_grid

    @property
    def interpolator_with_extrapolation(self):
        """
        Creates an RegularGridInterpolator (see scipy.interpolate) in the LocalCoord frame using the self.grid_tensor
        entries as data.
        """
        if self._interpolator_with_extrapolation is None:
            x = np.arange(0, self.x_length, self.x_res)
            y = np.flip(np.arange(0, self.y_length, self.y_res))
            # flipped as local coord frame y-axis points in different direction than array frame row axis
            if self.dimension == 3:
                z = np.arange(0, self.z_length, self.z_res)
                # grid tensor is transposed (x and y swapped) to fit the x y z access order,
                # as y equals the row-axis (first axis) of the array frame
                self._interpolator_with_extrapolation = RegularGridInterpolator((x, y, z),
                                                                                np.transpose(self.grid_tensor, (1, 0, 2)),
                                                                                method="linear",
                                                                                bounds_error=False,
                                                                                fill_value=None)
            elif self.dimension == 2:
                # grid tensor is transposed (x and y swapped) to fit the x y z access order,
                # as y equals the row-axis (first axis) of the array frame
                self._interpolator_with_extrapolation = RegularGridInterpolator((x, y),
                                                                                np.transpose(self.grid_tensor[:, :, 0], (1, 0)),
                                                                                method="linear",
                                                                                bounds_error=False,
                                                                                fill_value=None)
        return self._interpolator_with_extrapolation

    @property
    def interpolator_with_bound_error(self):
        """
        Creates an RegularGridInterpolator (see scipy.interpolate) in the LocalCoord frame using the self.grid_tensor
        entries as data.
        """
        if self._interpolator_with_bound_error is None:
            x = np.arange(0, self.x_length, self.x_res)
            y = np.flip(np.arange(0, self.y_length, self.y_res))
            # flipped as local coord frame y-axis points in different direction than array frame row axis
            if self.dimension == 3:
                z = np.arange(0, self.z_length, self.z_res)
                # grid tensor is transposed (x and y swapped) to fit the x y z access order,
                # as y equals the row-axis (first axis) of the array frame
                self._interpolator_with_bound_error = RegularGridInterpolator((x, y, z),
                                                                                np.transpose(self.grid_tensor,
                                                                                             (1, 0, 2)),
                                                                                method="linear",
                                                                                bounds_error=True,
                                                                                fill_value=None)
            elif self.dimension == 2:
                # grid tensor is transposed (x and y swapped) to fit the x y z access order,
                # as y equals the row-axis (first axis) of the array frame
                self._interpolator_with_bound_error = RegularGridInterpolator((x, y),
                                                                                np.transpose(self.grid_tensor[:, :, 0],
                                                                                             (1, 0)),
                                                                                method="linear",
                                                                                bounds_error=True,
                                                                                fill_value=None)
        return self._interpolator_with_bound_error

    def _get_values_from_local_coords(self, local_coords: List[LocalCoord]) -> np.ndarray:
        array_coords = self.array_coords_from_local_coords(local_coords)
        return self.get_values_from_array_coords(array_coords)

    def _get_value_from_local_coord(self, local_coord: LocalCoord) -> float:
        array_coord = self.array_coord_from_local_coord(local_coord)
        return self.get_value_from_array_coord(array_coord)

    def get_interpolated_values_from_local_coords(self, local_coords: List[LocalCoord]) -> np.ndarray:
        """In out of bound this function extrapolates."""
        row_vectors = np.vstack([local_coord.as_array() for local_coord in local_coords])
        interpolator = self.interpolator_with_extrapolation
        if self.dimension == 2:
            assert row_vectors[:, -1].any() == 0.0, "For assessing values in two-dimensional array, set z = 0"
            return interpolator(row_vectors[:, 0:2])
        return interpolator(row_vectors)

    def get_value_from_local_coord(self, local_coord: LocalCoord) -> float:
        """In out of bound this funtion falls back to get_value_from_array_coord to be able to get a value at the edges of the operation space."""
        try:
            interpolator = self.interpolator_with_bound_error
            if self.dimension == 2:
                assert local_coord.z == 0.0, "For assessing values in two-dimensional array, set z = 0"
                return interpolator(local_coord.as_array()[0:2]).item()
            elif self.dimension == 3:
                return interpolator(local_coord.as_array()).item()
        except ValueError:  # out of bound error from interpolation near grid bounds, fal back to get_value from array coord
            return self._get_value_from_local_coord(local_coord)

    def get_values_from_array_coords(self, array_coords: List[ArrayCoord]) -> np.ndarray:
        row_vectors = np.vstack([array_coord.as_array() for array_coord in array_coords])
        row_indices = row_vectors[:, 0]
        column_indices = row_vectors[:, 1]
        layer_indices = row_vectors[:, 2]
        return self.grid_tensor[row_indices, column_indices, layer_indices]

    def get_value_from_array_coord(self, array_coord: ArrayCoord) -> float:
        return self.grid_tensor[array_coord.row, array_coord.col, array_coord.lay].item()

    def set_from_func(self, func):
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        xv, yv, zv = astuple(self.local_coord_meshgrid)
        self.grid_tensor = func(xv, yv, zv)

    def set_from_array(self, input_array: ndarray):
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        if self.dimension <= 3:
            assert self.grid_tensor.shape[0] == input_array.shape[0], "Grid dimensions do not match."
            assert self.grid_tensor.shape[1] == input_array.shape[1], "Grid dimensions do not match."
        elif self.dimension == 3:
            assert self.grid_tensor.shape[2] == input_array.shape[2], "Grid dimensions do not match."
        if self.dimension == 3:
            self.grid_tensor = cp.deepcopy(input_array)
        elif self.dimension == 2:
            self.grid_tensor[:, :, self.main_layer] = cp.deepcopy(input_array)

    def add_from_array(self, input_array: ndarray, operation_type="add"):
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        assert self.grid_tensor.shape == input_array.shape, "Array dimensions do not match."
        if operation_type == "add":
            self.grid_tensor += input_array
        elif operation_type == "min":
            self.grid_tensor = np.minimum(self.grid_tensor, input_array)
        elif operation_type == "max":
            self.grid_tensor = np.maximum(self.grid_tensor, input_array)
        else:
            assert False, "Operator not known."

    def add_from_func(self, func, operation_type="add"):
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

        xv, yv, zv = astuple(self.local_coord_meshgrid)
        if operation_type == "add":
            self.grid_tensor += func(xv, yv, zv)
        elif operation_type == "min":
            self.grid_tensor = np.minimum(self.grid_tensor, func(xv, yv, zv))
        elif operation_type == "max":
            self.grid_tensor = np.maximum(self.grid_tensor, func(xv, yv, zv))
        else:
            assert False, "Operator not known."

    def set_local_coord_to_value(self, local_coord: LocalCoord, value: float):
        array_coord = self.array_coord_from_local_coord(local_coord)
        self.set_array_coord_to_value(array_coord, value)
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    def set_array_coord_to_value(self, array_coord: ArrayCoord, value: float):
        self.grid_tensor[array_coord.row, array_coord.col, array_coord.lay] = value
        self._interpolator_with_extrapolation = None
        self._interpolator_with_bound_error = None

    def plot_layer_flat(self, layer=None, colormap="cividis", norm_min_max: Optional[Tuple[float]] = None, title=None,
                        savepath='', colbar=False, circles=None, legend=None, plot_lib="plotly"):
        """Plots the grid_map using specified plot_lib."""
        config = get_config()
        if config.suppress_grid_image_plot and config.suppress_grid_image_save:
            return
        else:
            if plot_lib == "matplotlib":
                self._plot_layer_flat_matplotlib(layer=layer, colormap=colormap, norm_min_max=norm_min_max, title=title,
                                                 savepath=savepath, colbar=colbar, circles=circles, legend=legend)
            elif plot_lib == "plotly":
                self._plot_layer_flat_plotly(layer=layer, colormap=colormap, norm_min_max=norm_min_max, title=title,
                                             savepath=savepath, colbar=colbar, circles=circles, legend=legend)

    def _plot_layer_flat_plotly(self, layer=None, colormap="Cividis", norm_min_max: Optional[Tuple[float]] = None,
                                title=None, savepath='', colbar=False, circles=None, legend=None, show_figure=True, close=True):
        """Plots the grid_map using plotly."""
        config = get_config()
        if layer is None:
            layer = self.main_layer

        # Extract grid data for the selected layer
        grid_data = self.grid_tensor[:, :, layer]

        # Normalize values if min-max is provided
        if norm_min_max is None:
            zmin, zmax = float(np.min(grid_data)), float(np.max(grid_data))
        else:
            zmin, zmax = norm_min_max

        extent = (0, self.x_length, 0, self.y_length)

        # Initialize the figure
        fig = go.Figure()

        # Plot the heatmap (similar to imshow in matplotlib)
        fig.add_trace(go.Heatmap(
            x= np.arange(0, self.x_length, self.x_res),
            y = np.arange(0, self.y_length, self.y_res),
            z = np.flip(grid_data, axis=0),
            colorscale=colormap,
            zmin=zmin,
            zmax=zmax#,
            #colorbar=dict(title="Scale") if colbar else None,
        ))

        # Plot circles (if provided)
        if circles is not None:
            circle_x = []
            circle_y = []
            for i in range(len(circles)):
                x, y = circles[i][:2]  # Ignore the third component (z) if provided
                circle_x.append(x)
                circle_y.append(y)
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='markers',
                marker=dict(size=self.x_length / 100, color='gray', symbol='circle'),
                name="Circles"
            ))

        # Set axis labels
        fig.update_layout(
            xaxis_title='x Position (m)',
            yaxis_title='y Position (m)',
            title=title if title else '',
            width=2080,
            height=2080,
            xaxis=dict( tickfont=dict(size=60), range=[0.0, self.x_length - self.x_res], title_standoff=70, dtick=200),
            yaxis=dict(tickfont=dict(size=60), range=[0.0, self.y_length - self.y_res], title_standoff=70, dtick=200),  #scaleanchor="x",
            xaxis_title_font=dict(size=60),  # Change the size as needed
            yaxis_title_font=dict(size=60),  # Change the size as needed
            margin=dict(l=20, r=20, t=20, b=20)  # left, right, top, bottom margins
        )



        # Add a legend if specified
        if legend is not None:
            fig.update_layout(showlegend=True)

        fig = _save_and_close_plotly(close, fig, savepath, show_figure)
        return fig

    def _plot_layer_flat_matplotlib(self, layer=None, colormap="cividis", norm_min_max: Optional[Tuple[float]] = None,
                                    title=None, savepath='', colbar=False, circles=None, legend=None, close=True, show_figure=True):
        config = get_config()
        if layer is None:
            layer = self.main_layer
        colors = get_colors()
        # if config.save_as_pdf and config.suppress_image_plot:
        #     # use LaTeX fonts in the plot
        #     plt.rc('text', usetex=True)
        #     plt.rc('font', family='serif', size=12)
        extent = (0, self.x_length, -0, self.y_length)
        if norm_min_max is None:
            norm = mpl.colors.Normalize(vmin=float(np.min(self.grid_tensor[:, :, layer])),
                                        vmax=float(np.max(self.grid_tensor[:, :, layer])))
        else:
            norm = mpl.colors.Normalize(vmin=norm_min_max[0], vmax=norm_min_max[1])
        plt.imshow(self.grid_tensor[:, :, layer], origin='upper', extent=extent, norm=norm,
                   cmap=colormap,
                   interpolation='nearest')
        plt.xlabel('x Position (m)')
        plt.ylabel('y Position (m)')
        if colbar:
            plt.colorbar()
        if circles is not None:
            ax = plt.gca()
            if len(circles[0]) == 3:
                for i in range(0, len(circles), 2):
                    # col = colors[i%len(colors)]
                    col = "gray"
                    x, y, z = circles[i]
                    ax.add_patch(plt.Circle((x, y), self.x_length / 100, color=col))
                    x, y, z = circles[i + 1]
                    ax.add_patch(plt.Circle((x, y), self.x_length / 100, color=col))
            else:
                for i in range(0, len(circles)):
                    # col = colors[i%len(colors)]
                    col = "gray"
                    x, y = circles[i]
                    ax.add_patch(plt.Circle((x, y), self.x_length / 100, color=col))
        if legend is not None:
            ax = plt.gca()
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
            ax.legend(*legend, loc="lower center", fontsize="small", ncol=3,
                      bbox_to_anchor=(0.5, 1))  # loc= "center left", bbox_to_anchor=(1.05, 0.5)
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        _save_and_close_matplotlib(close, savepath, show_figure)

    def save_layer_landscape_animation(self, layer=None, colormap="cividis", title=None, savepath=''):
        print(f"Animating video saving to {savepath}.")
        if layer is None:
            layer = self.main_layer
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        def init():
            self.__plot_layer_landscape_matplotlib(fig=fig, ax=ax, layer=layer, colormap=colormap)
            ax.view_init(elev=30, azim=0)  # Change azimuth angle for each frame
            return fig,

        def animate(i):
            ax.view_init(elev=30., azim=i)
            return fig,

        ani = animation.FuncAnimation(fig, animate,
                                      frames=np.hstack(
                                          (np.arange(0, 92, 2), np.flip(np.arange(2, 90, 2)), np.arange(0, 92, 2))),
                                      init_func=init, interval=20, blit=True)

        # Save the animation as a video file (MP4) or GIF
        ani.save(os.path.join(savepath, f"{self.name}_layer_{layer}.mp4"), writer='ffmpeg', fps=30)

    def _plot_layer_landscape_matplotlib(self, layer=None, colormap="cividis", title=None, savepath='', colbar=False,
                             show_image=True, close=True):
        config = get_config()
        extent = (0, self.x_length, -0, self.y_length, 0, self.z_length)
        self.__plot_layer_landscape_matplotlib(layer=layer, colormap=colormap)
        if colbar:
            plt.colorbar()
        if title is not None:
            plt.title(title)

        _save_and_close_matplotlib(close, savepath, show_image)

    def _plot_layer_landscape_plotly(self,  layer=None, colormap="cividis", title=None, savepath='', colbar=False,
                             show_image=True, close=True, z_axis_title=None):
        if layer is None:
            layer = self.main_layer
        xv, yv, zv = astuple(self.local_coord_meshgrid)
        X = xv[:, :, layer]
        Y = yv[:, :, layer]
        Z = self.grid_tensor[:, :, layer]
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colormap)])
        fig.update_layout(
            scene=dict(
                xaxis_title='x Position (m)',
                yaxis_title='y Position (m)',
                zaxis_title=z_axis_title,
                aspectmode="cube"
            ),
            title=title if title else '',
        )

        # Show color bar if required
        if colbar:
            fig.update_layout(coloraxis_colorbar=dict(title='Color Scale'))
        fig = _save_and_close_plotly(close, fig, savepath, show_image)
        return fig


    def plot_layer_landscape(self, layer=None, colormap="cividis", title=None, savepath='', colbar=False,
                             plot_lib="plotly", z_axis_title=None):
        config = get_config()
        if config.suppress_grid_image_plot and config.suppress_grid_image_save:
            return
        else:
            if plot_lib == "matplotlib":
                self._plot_layer_landscape_matplotlib(layer=layer, colormap=colormap, title=title,savepath=savepath,colbar=colbar, z_axis_title=z_axis_title)
            elif plot_lib == "plotly":
                self._plot_layer_landscape_plotly(layer=layer, colormap=colormap, title=title,savepath=savepath,colbar=colbar, z_axis_title=z_axis_title)
            else:
                assert False, "Unknown plot_lib"



    def __plot_layer_landscape_matplotlib(self, fig=None, ax=None, layer=None, colormap='cividis', z_axis_title=None):
        if layer is None:
            layer = self.main_layer
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.axes(projection='3d')

        xv, yv, zv = astuple(self.local_coord_meshgrid)
        X = xv[:, :, layer]
        Y = yv[:, :, layer]
        Z = self.grid_tensor[:, :, layer]
        ax.plot_surface(X, Y, Z, norm=mpl.colors.Normalize(), cmap=colormap)
        ax.set_xlabel('x Position (m)')
        ax.set_ylabel('y Position (m)')
        ax.set_zlabel(z_axis_title)
        return fig, ax

    def plot_volume(self, save_path="", title=""):
        config = get_config()
        if config.suppress_grid_image_plot and config.suppress_grid_image_save:
            return
        xv, yv, zv = astuple(self.local_coord_meshgrid)
        fig = go.Figure(data=go.Volume(
            x=xv.flatten(),
            y=yv.flatten(),
            z=zv.flatten(),
            value=self.grid_tensor.flatten(),  # Flatten the 3D array for the plot
            isomin=np.min(self.grid_tensor).astype(float),
            isomax=np.max(self.grid_tensor).astype(float),
            opacity=0.1,  # Adjust the transparency
            surface_count=20  # Number of isosurfaces
        ))
        if title is not None:
            fig.update_layout(title=title)
        _save_and_close_plotly(True, fig, save_path, True)


    def plot_slices(self, save_folder, colormap="cividis", title=None, colbar=False, circles=None, legend=None):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        max_val = np.max(self.grid_tensor)
        min_val = np.min(self.grid_tensor)
        map_specific_folder = os.path.join(save_folder, self.name)
        if not os.path.exists(map_specific_folder):
            os.makedirs(map_specific_folder)
        for layer in range(self.shape[2]):
            savepath = os.path.join(map_specific_folder, f"{self.name}_layer{str(layer).zfill(4)}")
            self.plot_layer_flat(layer=layer, colormap=colormap, norm_min_max=(min_val, max_val), title=title,
                                 savepath=savepath, colbar=colbar, circles=circles, legend=legend, plot_lib="plotly")

    def _plot_paths_matplotlib(self, paths: Union[List[Path], Path], savepath='', suffix='', title='', prefix="", linestyle='',
                               markerstyle='.'):
        config = get_config()
        # plot map first
        self._plot_layer_flat_matplotlib(title=title, close=False, show_figure=False)
        ax = plt.gca()
        if isinstance(paths, Path):
            paths = [paths]
        colors = get_colors()
        for idx, path in enumerate(paths):
            vec_x = path.vec_x
            vec_y = path.vec_y
            ax.plot(vec_x, vec_y, color=colors[idx % len(colors)], marker=markerstyle, linestyle=linestyle,
                     markersize=2)  # will be shown in the plot() function
        if savepath != "":
            savepath = os.path.join(savepath, f'{prefix}opt_path_on_{self.name}{suffix}')
        _save_and_close_matplotlib(True, savepath, True)

    def _plot_paths_plotly(self, paths: Union[List[Path], Path], savepath='', suffix='', title='', prefix="", linestyle='dot',
                               markerstyle='.'):
        fig = self._plot_layer_flat_plotly(title=title, close=False, show_figure=False)
        if isinstance(paths, Path):
            paths = [paths]
        colors = get_colors(plot_lib="plotly")
        for idx, path in enumerate(paths):
            vec_x = path.vec_x
            vec_y = path.vec_y
            fig.add_trace(go.Scatter(
                x=vec_x,
                y=vec_y,
                mode='markers+lines',  # Show both markers and lines
                marker=dict(
                    color=colors[idx % len(colors)],  # Color from the colors list
                    size=2,  # Size of the markers
                    symbol=markerstyle  # Marker style (you can customize this)
                ),
                line=dict(
                    color=colors[idx % len(colors)],  # Line color (can be different or the same as markers)
                    dash=linestyle  # Line style (e.g., 'solid', 'dash', 'dot')
                )
            ))

        if savepath != "":
            savepath = os.path.join(savepath, f'{prefix}opt_path_on_{self.name}{suffix}')
        _save_and_close_plotly(True, fig, savepath, True)


    def plot_paths(self, paths: Union[List[Path], Path], savepath='', suffix='', title='', prefix="", linestyle='',
                   markerstyle='.', plot_lib="plotly"):
        """Plots a grid grid_map and overlays it with the plot of a path given by its coordinates vec_x and vec_y."""
        config = get_config()
        if config.suppress_grid_image_plot and config.suppress_grid_image_save:
            return
        else:
            if plot_lib == "matplotlib":
                self._plot_paths_matplotlib(paths=paths, savepath=savepath, suffix=suffix, title=title, prefix=prefix, linestyle=linestyle, markerstyle=markerstyle)
            elif plot_lib == "plotly":
                linestyle = 'dot'
                markerstyle = 'circle-dot'
                self._plot_paths_plotly(paths=paths, savepath=savepath, suffix=suffix, title=title, prefix=prefix, linestyle=linestyle, markerstyle=markerstyle)
            else:
                assert False, "Unknown plot_lib"



    def plot_3D_paths(self, paths: Union[List[Path], Path], savepath='', suffix="", prefix="", plot_lib="plotly"):
        """Plots a 3D grid grid_map and overlays it with the plot of a path given by its coordinates vec_x and vec_y."""
        config = get_config()
        if config.suppress_grid_image_plot and config.suppress_grid_image_save:
            return
        else:
            if plot_lib == "matplotlib":
                self._plot_3D_paths_matplotlib(paths=paths, savepath=savepath, suffix=suffix, prefix=prefix)
            elif plot_lib == "plotly":
                self._plot_3D_paths_plotly(paths=paths, savepath=savepath, suffix=suffix, prefix=prefix)
            else:
                assert False, "Unknown plot_lib"


    def _plot_3D_paths_matplotlib(self, paths: Union[List[Path], Path], savepath='', suffix='', prefix=''):
        self._plot_layer_landscape_matplotlib(close=False, show_image=False)
        ax = plt.gca()
        if isinstance(paths, Path):
            paths = [paths]
        colors = get_colors()
        for idx, path in enumerate(paths):
            ax.plot(path.vec_x, path.vec_y, path.vec_z, color=colors[idx % len(colors)])

        if savepath != "":
            savepath = os.path.join(savepath, f'{prefix}opt_path_on_{self.name}{suffix}')
        _save_and_close_matplotlib(True, savepath, True)


    def _plot_3D_paths_plotly(self, paths: Union[List[Path], Path], savepath='', suffix='', prefix=''):
        fig = self._plot_layer_landscape_plotly(close=False, show_image=False)
        if isinstance(paths, Path):
            paths = [paths]
        colors = get_colors(plot_lib="plotly")
        for idx, path in enumerate(paths):
            fig.add_trace(go.Scatter3d(
                x=path.vec_x,
                y=path.vec_y,
                z=path.vec_z,
                mode='lines',  # Use 'lines' to draw the path
                line=dict(
                    color=colors[idx % len(colors)],  # Set the line color
                    width=4  # You can adjust the width as needed
                )
            ))

            # Optional: Update the layout if needed (like adding axis labels)
            fig.update_layout(
                scene=dict(
                    xaxis_title='x Position (m)',
                    yaxis_title='y Position (m)',
                    zaxis_title='z Position (m)'
                ),
                title='3D Path Plot'
            )

        if savepath != "":
            savepath = os.path.join(savepath, f'{prefix}opt_path_on_{self.name}{suffix}')
        _save_and_close_plotly(True, fig, savepath, True)

    def export_lon_lat_height_pd_pkl(self, save_path: str, save_name=None):
        longitudes = []
        latitutes = []
        heights = []
        values = []
        for row, col in itertools.product(range(self.shape[0]), range(self.shape[1])):
            array_coord = ArrayCoord(row, col, 0)
            local_coord = self.local_coord_from_array_coord(array_coord)
            tmerc_coord = self.tmerc_coord_from_local_coord(local_coord)
            global_coord = global_coord_from_tmerc_coord(tmerc_coord)
            lon = global_coord.lon
            lat = global_coord.lat
            for idx, val in enumerate(self.grid_tensor[row, col, :]):
                height = idx*self.z_res
                longitudes.append(lon)
                latitutes.append(lat)
                heights.append(height)
                values.append(val)

        data = {
            'lon': longitudes,
            'lat': latitutes,
            'height': heights,
            'value': values
        }

        df = pd.DataFrame(data)

        # Reset the index to remove it and prevent it from being saved
        df = df.reset_index(drop=True)

        # Save the DataFrame as a pickle file without the index
        if ".pkl" in save_path:
            df.to_pickle(save_path)
        else:
            df.to_pickle(os.path.join(save_path, save_name))



