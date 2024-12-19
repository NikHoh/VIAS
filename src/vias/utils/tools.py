import matplotlib.colors as matlib_col
import numpy as np
import plotly.colors as plotly_col
from matplotlib import pyplot as plt

from vias.config import get_config


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def euclidian_distance(p0, p1):
    return float(np.linalg.norm(np.array(p0) - np.array(p1)))


def float_2_str(value: float) -> str:
    return f"{value:.3f}".replace(".", "_")


def get_polygon_area(polygon: list):
    """Calculates the area of a polygon by the cross product method."""
    unzipped = list(zip(*polygon, strict=False))
    x = unzipped[0]
    y = unzipped[1]
    if len(x) < 3:
        return 0
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_colors(plot_lib="matplotlib"):
    colors = []
    color_names = [
        "lime green",
        "cyan",
        "orange",
        "red",
        "magenta",
        "warm brown",
        "gold",
        "grey",
        "black",
        "silver",
        "emerald",
        "clear blue",
        "raspberry",
        "electric purple",
        "pale pink",
        "ocean",
        "pale green",
        "pinkish orange",
        "sandy yellow",
        "terracotta",
        "slime green",
        "stormy blue",
        "vomit",
        "topaz",
        "flat blue",
        "seaweed",
        "medium purple",
        "dark fuchsia",
    ]
    for name in color_names:
        colors.append(
            matlib_col.to_rgba(matlib_col.get_named_colors_mapping()[f"xkcd:{name}"])
        )
    if plot_lib == "matplotlib":
        return colors
    elif plot_lib == "plotly":
        return [plotly_col.label_rgb(np.array(color[0:3]) * 255) for color in colors]
    else:
        raise AssertionError("Unknown plot_lib")


def get_linestyles():
    linestyles = [
        "solid",
        (0, (1, 1)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (5, 1)),  # densely dashed
        (0, (3, 5, 1, 5)),  # dashdotted
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
        (0, (5, 10)),  # loosely dashed
        (0, (3, 10, 1, 10, 1, 10)),  # loosely dashdotdotted
        (0, (3, 10, 1, 10)),  # loosely dashdotted
        (0, (1, 10)),
    ]  # loosely dotted
    return linestyles


def get_markers():
    return [".", "1", "2", "3", "4", "+", "x", "*"]


def get_equally_points_between(
    p1: np.ndarray, p2: np.ndarray, max_distance: float
) -> list[np.ndarray]:
    total_distance = np.linalg.norm(p2 - p1)

    # Determine the number of intermediate points
    num_points = int(np.ceil(total_distance / max_distance)) - 1

    # Generate equally spaced points along the line
    points = [p1 + (p2 - p1) * (i / (num_points + 1)) for i in range(1, num_points + 1)]

    return [p1] + points + [p2]


def get_specific_number_of_points_between(
    p1: np.ndarray, p2: np.ndarray, num_points: int
) -> list[np.ndarray]:
    points = [p1 + (p2 - p1) * (i / (num_points + 1)) for i in range(1, num_points + 1)]
    assert len(points) == num_points
    return points


def _save_and_close_plotly(close, fig, savepath, show_figure):
    config = get_config()
    # Save the plot if needed
    if savepath != "" and not config.suppress_grid_image_save:
        fig.write_image(savepath + ".png")
        fig.write_html(savepath + ".html")
        # fig.write_json(savepath + '.json')
        if config.save_as_pdf:
            fig.write_image(
                savepath + ".pdf"
            )  # Plotly supports SVG, but not EPS directly
    # Show the image if requested
    if not config.suppress_grid_image_plot and show_figure:
        fig.show()
    if close:
        fig = None
    return fig


def _save_and_close_matplotlib(close, savepath, show_figure):
    config = get_config()
    if savepath != "" and not config.suppress_grid_image_save:
        plt.savefig("".join([savepath, ".png"]), bbox_inches="tight")
    if config.save_as_pdf:
        plt.savefig("".join([savepath, ".pdf"]), bbox_inches="tight")
    if not config.suppress_grid_image_plot and show_figure:
        plt.show()
    if close:
        plt.close()


def get_num_nodes(columns, layers, rows):
    return rows * columns * layers


def calculate_needed_graph_storage(columns, layers, rows):
    needed_storage = (
        (
            4 * get_num_nodes(columns, layers, rows)
            + 2 * get_num_edges(columns, layers, rows, True)
        )
        * 8
        / 1e6
    )  # in MB
    return needed_storage


def get_num_edges(columns, layers, rows, directed):
    # assuming a bidirectional 26 neighborhood
    num_expected_edges = (
        ((layers - 2) * (rows - 2) * (columns - 2) * 26)  # inner nodes
        + (
            (layers - 2) * (rows - 2) * 2
            + (layers - 2) * (columns - 2) * 2
            + (rows - 2) * (columns - 2) * 2
        )
        * 17  # outer plane nodes
        + ((layers - 2) * 4 + (rows - 2) * 4 + (columns - 2) * 4) * 11  # edge nodes
        + 8 * 7
    )  # corner nodes
    if not directed:
        return int(num_expected_edges / 2)
    return num_expected_edges
