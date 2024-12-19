import os
from enum import Enum

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import plotly.colors as plotly_col
import plotly.graph_objects as go
from pandas import DataFrame

from vias.utils.tools import get_colors, get_markers


class PlotBackend(Enum):
    MATPLOTLIB = 1
    PLOTLY = 2

    def __eq__(self, other):
        return self.value == other.value


def do_three_axis_pareto_plot(
    prefixes: list[str],
    dfs: list[DataFrame],
    plot_backend,
    savepath=None,
    show=True,
    pdf_save=False,
):
    assert isinstance(prefixes, list)
    assert isinstance(dfs, list)
    assert len(prefixes) == len(dfs)
    for prefix_idx, prefix in enumerate(prefixes):
        if prefix_idx == 0:
            title_string = f"{prefix} vs. "
        elif prefix_idx == len(prefixes) - 1:
            title_string += f"{prefix}"
        else:
            title_string += f"{prefix} vs. "
    colors = get_colors()
    plotly_colors = [
        plotly_col.label_rgb(np.array(color[0:3]) * 255) for color in colors
    ]
    plotly_markers = ["circle", "x", "square", "diamond", "cross"]
    x_string = "edge cost"
    y_string = "travel cost"
    z_string = "social cost"
    if plot_backend == PlotBackend.PLOTLY:
        fig = go.Figure()
    for df_idx, (prefix, df) in enumerate(zip(prefixes, dfs, strict=False)):
        df = df.rename(
            columns=dict(
                zip(
                    df.columns.values,
                    [prefix + tag for tag in df.columns.values],
                    strict=False,
                )
            ),
            inplace=False,
        )
        if plot_backend == PlotBackend.PLOTLY:
            fig.add_trace(
                go.Scatter3d(
                    x=df[prefix + x_string],
                    y=df[prefix + y_string],
                    z=df[prefix + z_string],
                    name=prefix,
                    mode="markers",
                    marker=dict(
                        symbol=plotly_markers[df_idx],
                        color=plotly_colors[df_idx],
                        opacity=0.5,
                    ),
                )
            )

        elif plot_backend == PlotBackend.MATPLOTLIB:
            print("3 axis plot not yet defined for Matplotlib")
            return
    if plot_backend == PlotBackend.MATPLOTLIB:
        pass  # ax.legend()  # plt.title(title_string)  # plt.xlabel(x_string)  #  #
        # plt.ylabel(y_string)  # plt.show()
    elif plot_backend == PlotBackend.PLOTLY:
        fig.update_layout(
            title=dict(
                text=title_string + f" {x_string} and {y_string} and {z_string}"
            ),
            scene=dict(
                xaxis_title=x_string, yaxis_title=y_string, zaxis_title=z_string
            ),
        )
        if savepath is not None:
            fig.write_html(
                os.path.join(
                    savepath,
                    f"{title_string}_{x_string}_{y_string}_" f"{z_string}.html",
                )
            )
            if pdf_save:
                fig.write_image(
                    os.path.join(
                        savepath,
                        f"{title_string}_{x_string}_{y_string}_" f"{z_string}.pdf",
                    )
                )
        if show:
            fig.show()


def do_two_axis_pareto_plot(
    prefixes: list[str],
    dfs: list[DataFrame],
    x_string,
    y_string,
    plot_backend,
    savepath=None,
    show=True,
    pdf_save=False,
    mark_knee=False,
):
    assert isinstance(prefixes, list)
    assert isinstance(dfs, list)
    assert len(prefixes) == len(dfs)
    colors = get_colors()
    markers = get_markers()
    plotly_colors = [
        plotly_col.label_rgb(np.array(color[0:3]) * 255) for color in colors
    ]
    plotly_markers = ["circle", "x", "square", "diamond", "cross"]
    for prefix_idx, prefix in enumerate(prefixes):
        if prefix_idx == 0:
            title_string = f"{prefix}"
        else:
            title_string += f" vs. {prefix}"
    if plot_backend == PlotBackend.PLOTLY:
        fig = go.Figure()
    elif plot_backend == PlotBackend.MATPLOTLIB:
        legend_handles = []
        labels = []
        fig, ax = plt.subplots(1)
        ax.set(xlabel=x_string, ylabel=y_string)
    knee_points = {}
    for df_idx, (prefix, df) in enumerate(zip(prefixes, dfs, strict=False)):
        if x_string not in df.columns.values:
            print(f"{x_string} not in df with id {df_idx}")
            continue
        if y_string not in df.columns.values:
            print(f"{y_string} not in df with id {df_idx}")
            continue
        df = df.rename(
            columns=dict(
                zip(
                    df.columns.values,
                    [prefix + tag for tag in df.columns.values],
                    strict=False,
                )
            ),
            inplace=False,
        )
        df.sort_values(prefix + x_string, inplace=True)

        if mark_knee:
            unnorm_pareto_set = np.stack(
                (
                    np.array(df[prefix + "edge cost"]),
                    np.array(df[prefix + "travel cost"]),
                    np.array(df[prefix + "social cost"]),
                ),
                axis=1,
            )
            # norm pareto set (like it is done before knee point retrival)
            norm_pareto_set = (
                unnorm_pareto_set - np.min(unnorm_pareto_set, axis=0)
            ) / (np.max(unnorm_pareto_set, axis=0) - np.min(unnorm_pareto_set, axis=0))
            knee_points[prefix] = dict(
                zip(
                    ["edge cost", "travel cost", "social cost"],
                    list(
                        unnorm_pareto_set[
                            np.argmin(np.linalg.norm(norm_pareto_set, axis=1)), :
                        ]
                    ),
                    strict=False,
                )
            )

        if plot_backend == PlotBackend.MATPLOTLIB:
            color = colors[df_idx % len(colors)]
            marker = markers[df_idx % len(markers)]
            labels.append(prefix)
            ax.plot(
                df[prefix + x_string].values,
                df[prefix + y_string].values,
                c=color,
                linestyle="",
                marker=marker,
                label="_nolegend_",
            )
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker=marker, linestyle="")
            )
        elif plot_backend == PlotBackend.PLOTLY:
            fig.add_trace(
                go.Scatter(
                    x=df[prefix + x_string],
                    y=df[prefix + y_string],
                    name=prefix,
                    mode="markers",
                    marker=dict(
                        symbol=plotly_markers[df_idx],
                        color=plotly_colors[df_idx],
                        opacity=0.5,
                    ),
                )
            )

    if mark_knee and plot_backend == PlotBackend.MATPLOTLIB:
        for df_idx, (prefix, _) in enumerate(zip(prefixes, dfs, strict=False)):
            marker = markers[df_idx % len(markers) + 6]
            ax.plot(
                knee_points[prefix][x_string],
                knee_points[prefix][y_string],
                c="black",
                linestyle="",
                marker=marker,
                label="_nolegend_",
            )
            legend_handles.append(
                mlines.Line2D([], [], color="black", marker=marker, linestyle="")
            )
            labels.append(prefix + " knee point")

    if plot_backend == PlotBackend.MATPLOTLIB:
        ax.legend(legend_handles, labels)

        plt.xlabel(x_string)
        plt.ylabel(y_string)
        plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()
        if pdf_save:
            fig.savefig(
                os.path.join(savepath, f"{title_string}_{x_string}_{y_string}.pgf")
            )
        # plt.title(title_string + f" {x_string} and {y_string}")
        fig.savefig(os.path.join(savepath, f"{title_string}_{x_string}_{y_string}.pdf"))
        if show:
            plt.show()
    elif plot_backend == PlotBackend.PLOTLY:
        fig.update_xaxes(title=x_string)
        fig.update_yaxes(title=y_string)
        fig.update_layout(title=dict(text=title_string + f" {x_string} and {y_string}"))
        if savepath is not None:
            fig.write_html(
                os.path.join(savepath, f"{title_string}_{x_string}_{y_string}.html")
            )
            if pdf_save:
                fig.write_image(
                    os.path.join(savepath, f"{title_string}_{x_string}_{y_string}.pdf")
                )
        if show:
            fig.show()
