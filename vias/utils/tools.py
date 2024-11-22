import copy as cp
# from plotter import mlab
import itertools
import math
import random
from collections import Counter
from datetime import datetime as dt
from functools import lru_cache, reduce
from itertools import product

import geomdl
import matplotlib.colors as matlib_col
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import plotly.colors as plotly_col
# import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def order_of_magnitude(number):
    return math.floor(math.log(number, 10))


def euclidian_distance(p0, p1):
    return float(np.linalg.norm(np.array(p0) - np.array(p1)))
    # if len(p0) == 3:
    #     return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
    # elif len(p0) == 2:
    #     return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def cart2pol(x, y):
    """Converts 2D cartesian coordinates into polar coordinates."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """Converts 2D polar coordinates into cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def float_2_str(value: float) -> str:
    return f"{value:.3f}".replace(".", "_")


def get_polygon_area(polygon: list):
    """Calculates the area of a polygon by the crossproduct method."""
    unzipped = list(zip(*polygon))
    x = unzipped[0]
    y = unzipped[1]
    if len(x) < 3:
        return 0
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _get_polygon_area_v2(polygon: list):
    """Calculates the area of a polygon by the shoelace formula. Caution: this method can return a negative area
     dependent on the direction of the polygon's points
    defined. (negative, if points are clockwise defined). Only use this for the centroid method, as there the negative
    area is necessary to get positive centroid coordinates again.
    see: https://en.wikipedia.org/wiki/Centroid#cite_note-:0-19"""
    unzipped = list(zip(*polygon))
    x = unzipped[0]
    y = unzipped[1]
    n = len(x)
    if n < 3:
        assert 1 == 0, "Polygon is not an area."
    area = 0
    for i in range(0, n):
        area += (y[i] + y[(i + 1) % n]) * (x[i] - x[(i + 1) % n])
    return area / 2


def get_polygon_centroid(polygon: list):
    """Calculates the centroid of a polygon."""
    area = _get_polygon_area_v2(polygon)
    unzipped = list(zip(*polygon))
    x = unzipped[0]
    y = unzipped[1]
    n = len(x)
    x_centroid = 0
    y_centroid = 0
    for i in range(0, n):
        x_centroid += (x[i] + x[(i + 1) % n]) * (x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i])
        y_centroid += (y[i] + y[(i + 1) % n]) * (x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i])
    return x_centroid / (6 * area), y_centroid / (6 * area)


def get_nearest_point_on_polygon(p: tuple, polygon: list):
    """Calculates the nearest point on a polygon given a point p inside or outside the polygon. Returns the nearest
    point on the polygon and its distance to p. In the current implementation there is no differentiation if p lies in
    the polygon or outside. For mathematical background see
    https://gis.stackexchange.com/questions/104161/how-can-i-find-closest-point-on-a-polygon-from-a-point"""
    n = len(polygon)
    dist = []
    points = []
    for k in range(0, n):
        x_1 = polygon[k][0]
        y_1 = polygon[k][1]
        x_2 = polygon[(k + 1) % n][0]
        y_2 = polygon[(k + 1) % n][1]
        p_x = p[0]
        p_y = p[1]

        dx = x_2 - x_1
        dy = y_2 - y_1

        i = ((p_x - x_1) * dx + (p_y - y_1) * dy) / (dx ** 2 + dy ** 2)
        if i <= 0:
            nearest_x = x_1
            nearest_y = y_1
        elif i >= 1:
            nearest_x = x_2
            nearest_y = y_2
        else:
            nearest_x = x_1 + i * dx
            nearest_y = y_1 + i * dy
        points.append((nearest_x, nearest_y))
        dist.append(euclidian_distance((nearest_x, nearest_y), p))
    return points[dist.index(min(dist))], min(dist)


def draw_graphs(graphs, nodes, only_nodes=False):
    """
    Helper method for drawing networkx graphs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    func = nx.draw_networkx

    if only_nodes:
        func = nx.draw_networkx_nodes
    node_ids = [i for i in range(0, len(nodes))]
    node_dic = dict(zip(node_ids, [(x, y) for (x, y, z) in nodes]))
    colors = ['k', 'g', 'b', 'r', 'y']
    for i, graph in enumerate(graphs):
        func(graph, node_dic, node_size=25, with_labels=True, ax=ax, edge_color=colors[i])
    plt.show()
    plt.close()


def save_graphs(graphs, savepath, label: str, print_node_label, vertiport_nodes, only_nodes=False):
    """
    Helper method for drawing networkx graphs.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    func = nx.draw_networkx

    if only_nodes:
        func = nx.draw_networkx_nodes

    color_base = ['k', 'r', 'b', 'g', 'y']
    colors = []
    while len(colors) < len(graphs):
        colors.extend(color_base)
    for i, graph in enumerate(graphs):
        node_dic = nx.get_node_attributes(graph, 'pos')
        for key, val in node_dic.items():
            node_dic[key] = (val[0], val[1])
        if i == 0:
            if vertiport_nodes:
                node_size = 1
            else:
                node_size = 25
            func(graph, node_dic, node_size=node_size, with_labels=print_node_label, ax=ax, edge_color=colors[i],
                 label=label)
        else:
            func(graph, node_dic, node_size=1, with_labels=False, ax=ax, edge_color=colors[1])
    if vertiport_nodes:
        node_dic = nx.get_node_attributes(vertiport_nodes, 'pos')
        for key, val in node_dic.items():
            node_dic[key] = (val[0], val[1])
        nx.draw_networkx_nodes(vertiport_nodes, node_dic, node_size=25, ax=ax, node_color='g')
    plt.legend()
    plt.savefig(savepath)
    plt.close()


# def mirror_2d_point(mirror_point1: np.array, mirror_point2: np.array, point_to_mirror: np.array) -> np.array:
#     assert len(mirror_point1) == len(mirror_point2) == len(point_to_mirror) == 2
#     x1 = mirror_point1[0]
#     y1 = mirror_point1[1]
#     x2 = mirror_point2[0]
#     y2 = mirror_point2[1]
#     x3 = point_to_mirror[0]
#     y3 = point_to_mirror[1]
#
#     # normal vector on mirror
#     n = np.array([y1 - y2, x2 - x1])
#     if mirror_point1.T @ n >= 0:
#         n_0 = n / math.sqrt((y1 - y2) ** 2 + (x2 - x1) ** 2)
#     else:
#         n_0 = - n / math.sqrt((y1 - y2) ** 2 + (x2 - x1) ** 2)
#
#     # distance from point_to_mirror to mirror
#     d = (point_to_mirror - mirror_point1).T @ n_0  # dot product
#
#     return point_to_mirror - 2 * d * n_0

def test_2d_gauss_distribution():
    # 2 dim
    sigma_x = 5
    sigma_y = 5
    total_values = 10000000
    counter_huge_elipsoid = 0  # Elipsoid with axis length of 3 sigma respectively
    counter_big_elipsoid = 0  # Elipsoid with axis length of 3 sigma respectively
    counter_elipsoid = 0  # Elipsoid with axis length of 2 sigma respectively
    counter_small_elipsoid = 0  # Elipsoid with axis length of 1 sigma respectively

    for i in range(0, total_values):
        val_x = random.gauss(0, sigma_x)
        val_y = random.gauss(0, sigma_y)
        if (val_x / (2 * sigma_x)) ** 2 + (val_y / (2 * sigma_y)) ** 2 > 1:
            continue
        if (2 * val_x / (3 * sigma_x)) ** 2 + (2 * val_y / (3 * sigma_y)) ** 2 > 1:
            counter_huge_elipsoid += 1
            continue
        if (val_x / sigma_x) ** 2 + (val_y / sigma_y) ** 2 > 1:
            counter_huge_elipsoid += 1
            counter_big_elipsoid += 1
            continue
        if (2 * val_x / sigma_x) ** 2 + (2 * val_y / sigma_y) ** 2 > 1:
            counter_huge_elipsoid += 1
            counter_big_elipsoid += 1
            counter_elipsoid += 1
            continue
        else:
            counter_huge_elipsoid += 1
            counter_big_elipsoid += 1
            counter_elipsoid += 1
            counter_small_elipsoid += 1

    print("4 sigma Elipsoid: {}%".format(counter_huge_elipsoid / total_values * 100))
    print("3 sigma Elipsoid: {}%".format(counter_big_elipsoid / total_values * 100))
    print("2 sigma Elipsoid: {}%".format(counter_elipsoid / total_values * 100))
    print("1 sigma Elipsoid: {}%".format(counter_small_elipsoid / total_values * 100))


def test_1d_gauss_distribution():
    # 1 dim
    sigma = 5
    total_values = 1000000
    counter_1sigma = 0
    counter_2sigma = 0
    counter_3sigma = 0

    for i in range(0, total_values):
        val = random.gauss(0, sigma)
        if abs(val) > 3 * sigma:
            continue
        if abs(val) > 2 * sigma:
            counter_3sigma += 1
            continue
        if abs(val) > sigma:
            counter_3sigma += 1
            counter_2sigma += 1
            continue
        else:
            counter_3sigma += 1
            counter_2sigma += 1
            counter_1sigma += 1

    print("sigma: {}%".format(counter_1sigma / total_values * 100))
    print("2sigma: {}%".format(counter_2sigma / total_values * 100))
    print("3sigma: {}%".format(counter_3sigma / total_values * 100))


"""The following 3 functions are used to calculate proper divisors of a number n.
Code is taken from http://rosettacode.org/wiki/Proper_divisors#Python:_Functional"""

MUL = int.__mul__


def prime_factors(n):
    'Map prime factors to their multiplicity for n'
    d = _divs(n)
    d = [] if d == [n] else (d[:-1] if d[-1] == d else d)
    pf = Counter(d)
    return dict(pf)


@lru_cache(maxsize=None)
def _divs(n):
    'Memoized recursive function returning prime factors of n as a list'
    for i in range(2, int(math.sqrt(n) + 1)):
        d, m = divmod(n, i)
        if not m:
            return [i] + _divs(d)
    return [n]


def proper_divs(n):
    '''Return the set of proper divisors of n.'''
    pf = prime_factors(n)
    pfactors, occurrences = pf.keys(), pf.values()
    multiplicities = product(*(range(oc + 1) for oc in occurrences))
    divs = {reduce(MUL, (pf ** m for pf, m in zip(pfactors, multis)), 1)
            for multis in multiplicities}
    try:
        divs.remove(n)
    except KeyError:
        pass
    return divs or ({1} if n != 1 else set())


def get_nearest_div(n, nearest_to):
    divs = list(proper_divs(n))
    diffs = [abs(a - nearest_to) for a in divs]
    return divs[diffs.index(min(diffs))]


def get_angles_curve(obj):
    """Get list of all angles between the segemtns of a polygonal line.
    They are all positive"""
    angles = []
    if isinstance(obj, geomdl.NURBS.Curve):
        # raise geomdl.GeomdlException("Input shape must be an instance of abstract.Curve class")
        evalpts = [np.array(p) for p in obj.evalpts]
    else:
        evalpts = [np.array(p) for p in obj]

    assert evalpts[0].size == 3, "Not working with less or more than 3 dim"
    num_evalpts = len(evalpts)
    for idx in range(num_evalpts - 1):
        if idx == 0:
            continue
        # old way to calculate it (only positive)
        # a = euclidian_distance(evalpts[idx-1], evalpts[idx])
        # b = euclidian_distance(evalpts[idx], evalpts[idx+1])
        # B_to_C = evalpts[idx+1] - evalpts[idx]
        # A_shift = evalpts[idx-1] + B_to_C
        # c = euclidian_distance(A_shift, evalpts[idx])
        # argument = (a**2+ b**2 - c**2)/(2*a*b)
        # # assert argument >= -1.0 and argument <= 1.0, f"Cannot calc arccos with i-1: {evalpts[idx-1]}, i: {evalpts[idx]} and i+1: {evalpts[idx+1]}, a: {a}, b: {b}, B_to_c: {B_to_C}, A_shift: {A_shift}, c: {c}"
        # zaginess += np.arccos(max(-1.0, min(1.0, argument)))

        # another old way to calculate it (only positive)
        A_to_B = evalpts[idx] - evalpts[idx - 1]
        A_to_B = A_to_B / np.linalg.norm(A_to_B)
        # c = euclidian_distance(evalpts[idx], evalpts[idx -1])
        B_to_C = evalpts[idx + 1] - evalpts[idx]
        B_to_C = B_to_C / np.linalg.norm(B_to_C)
        # a = euclidian_distance(evalpts[idx],  evalpts[idx+1])
        argument = np.dot(A_to_B, B_to_C)
        angle1 = np.arccos(max(-1.0, min(1.0, argument)))

        # take another way to calculate angle with different signs
        V_ref = np.array([0, 0, 1])
        cross = np.cross(A_to_B, B_to_C)
        dot = np.dot(A_to_B, B_to_C)
        if np.all(cross == np.array([0, 0, 0])):
            angle2 = 0
        else:
            angle2 = np.arctan2(np.linalg.norm(cross), dot)
        test = np.dot(V_ref, cross)
        if test < 0.0:
            angle2 = -angle2
        # if abs(angle2) != angle1:
        #     print("Here")

        angles.append(angle2)
    return angles


def zaginess_curve(obj):
    """ Computes the zaginess (sum of angles between three consecutive points) of the curve.
    :param obj: input curve
    :type obj: abstract.Curve
    :return: zaginess
    :rtype: float
    """
    return sum(np.abs(get_angles_curve(obj)))


def angle_sum(obj):
    return sum(get_angles_curve(obj))


def variance_angle_curve(obj):
    """Computes the variance of angles between the segments of a polygonal path"""
    return np.std(get_angles_curve(obj)) ** 2


def angle_change_rate(obj):
    """Computes the central difference of the positive and negative angles summed up"""
    return sum(list(np.gradient(get_angles_curve(obj))))


def abs_angle_change_rate(obj):
    """Computes the central difference of the positive and negative angles summed up"""
    return sum(list(np.abs(np.gradient(get_angles_curve(obj)))))


def variance_angle_change_rate(obj):
    """Computes variance of the central difference of the positive and negative angles summed up"""
    return np.std(np.gradient(get_angles_curve(obj))) ** 2


def count_zero_crossings(obj):
    angles = get_angles_curve(obj)
    return len(np.where(np.diff(np.sign(angles)))[0])


def nx_plot_3D(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([list(G.degree())[i][1] for i in range(n)])

    # Define special_pos_color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

        # Set the initial view
        ax.view_init(30, angle)

        # Hide the axes
        ax.set_axis_off()

        plt.show()


# def nx_plot_3D_v2(H):
#     # reorder nodes from 0,len(G)-1
#     G = nx.convert_node_labels_to_integers(H)
#     # 3d spring layout
#     pos = nx.spring_layout(G, dim=3)
#     # numpy array of x,y,z positions in sorted node order
#     xyz = np.array([pos[v] for v in sorted(G)])
#     # scalar colors
#     scalars = np.array(list(G.nodes())) + 5
#
#     pts = mlab.points3d(
#         xyz[:, 0],
#         xyz[:, 1],
#         xyz[:, 2],
#         scalars,
#         scale_factor=0.1,
#         scale_mode="none",
#         colormap="Blues",
#         resolution=20,
#     )
#
#     pts.mlab_source.dataset.lines = np.array(list(G.edges()))
#     tube = mlab.pipeline.tube(pts, tube_radius=0.01)
#     mlab.pipeline.surface(tube, special_pos_color=(0.8, 0.8, 0.8))
#     mlab.show()


def get_distance_array(array: np.array):
    """Assuming a binary array (with 0 and x), calculating the distance_array containing the minimum distances
    in the array from cells with 0 to cells with x."""
    assert array.ndim == 2
    distance_array = np.zeros(array.shape)
    count = 0
    total_count = array.shape[0] * array.shape[1]
    for row, column in itertools.product(range(0, array.shape[0]), range(0, array.shape[1])):
        if array[row, column] != 0:
            distance_array[row, column] = 0
        else:
            if column - 1 >= 0:
                r_guess = distance_array[row, column - 1] - 5
            elif row - 1 >= 0:
                r_guess = distance_array[row - 1, column] - 5
            else:
                r_guess = 1
            if r_guess == np.nan:
                r = 1
            else:
                r = int(max(1, r_guess))
            while True:
                summed_circle_values = add_values_on_circle(array, row, column, r)
                if summed_circle_values != 0:
                    break
                r += 1
                if r > max(array.shape):
                    r = np.nan
                    break
                # print(f"\rr set to {r}.", end="")
            distance_array[row, column] = r
        print(f"\r{count} of {total_count} done.", end="")
        count += 1
    # care about nans
    distance_array = filter_nan(distance_array)
    return distance_array


def filter_nan(array):
    # mask = np.isnan(distance_array)
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    distance_array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~distance_array.mask]
    y1 = yy[~distance_array.mask]
    newdistance_array = distance_array[~distance_array.mask]

    GD1 = interpolate.griddata((x1, y1), newdistance_array.ravel(),
                               (xx, yy),
                               method='nearest')
    return GD1


def get_inverse_distance_array(array: np.array):
    """Assuming an binary array (with 0 and x), calculating the distance_array containing the minimum distances
    in the array from cells with x to cells with 0."""
    assert array.ndim == 2
    distance_array = np.zeros(array.shape)
    count = 0
    total_count = array.shape[0] * array.shape[1]
    for row, column in itertools.product(range(0, array.shape[0]), range(0, array.shape[1])):
        if array[row, column] == 0:
            distance_array[row, column] = 0
        else:
            if column - 1 >= 0:
                r_guess = distance_array[row, column - 1] - 5
            elif row - 1 >= 0:
                r_guess = distance_array[row - 1, column] - 5
            else:
                r_guess = 1
            if r_guess == np.nan:
                r = 1
            else:
                r = int(max(1, r_guess))
            while not zero_value_on_circle(array, row, column, r):
                r += 1
                # print(f"\rr set to {r}.", end="")
                if r > max(array.shape):
                    r = np.nan
                    break
            distance_array[row, column] = r
        print(f"\r{count} of {total_count} done.", end="")
        count += 1
    # care about nans
    distance_array = filter_nan(distance_array)
    return distance_array


def get_array_val(array, row, column, return_val):
    """Getting an array enry ignoring index errors."""
    try:
        assert type(row) == type(int())
        assert type(column) == type(int())
        if row >= 0 and column >= 0:
            return array[row, column]
        else:
            return return_val
    except IndexError:
        return return_val


def set_array_val(array, row, column, val):
    """Getting an array enry ignoring index errors."""

    assert type(row) == type(int())
    assert type(column) == type(int())
    if row >= 0 and column >= 0:
        array[row, column] = val


def add_values_on_circle(array, x0, y0, r):
    """Sums all values that lie on a circle with radius r around x0, y0 on an array. Algorithm: Midpoint circle algorithm."""
    summed_values = 0
    f = 1 - r
    ddf_x = 1
    ddf_y = -2 * r
    x = 0
    y = r
    summed_values += get_array_val(array, x0, y0 + r, 0)
    summed_values += get_array_val(array, x0, y0 - r, 0)
    summed_values += get_array_val(array, x0 + r, y0, 0)
    summed_values += get_array_val(array, x0 - r, y0, 0)

    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        summed_values += get_array_val(array, x0 + x, y0 + y, 0)
        summed_values += get_array_val(array, x0 - x, y0 + y, 0)
        summed_values += get_array_val(array, x0 + x, y0 - y, 0)
        summed_values += get_array_val(array, x0 - x, y0 - y, 0)
        summed_values += get_array_val(array, x0 + y, y0 + x, 0)
        summed_values += get_array_val(array, x0 - y, y0 + x, 0)
        summed_values += get_array_val(array, x0 + y, y0 - x, 0)
        summed_values += get_array_val(array, x0 - y, y0 - x, 0)
    return summed_values


def draw_circle(array, x0, y0, r, val):
    """Draws a circle with radius r around x0, y0 on an array. Algorithm: Midpoint circle algorithm."""
    f = 1 - r
    ddf_x = 1
    ddf_y = -2 * r
    x = 0
    y = r
    set_array_val(array, x0, y0 + r, val)
    set_array_val(array, x0, y0 - r, val)
    set_array_val(array, x0 + r, y0, val)
    set_array_val(array, x0 - r, y0, val)

    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        set_array_val(array, x0 + x, y0 + y, val)
        set_array_val(array, x0 - x, y0 + y, val)
        set_array_val(array, x0 + x, y0 - y, val)
        set_array_val(array, x0 - x, y0 - y, val)
        set_array_val(array, x0 + y, y0 + x, val)
        set_array_val(array, x0 - y, y0 + x, val)
        set_array_val(array, x0 + y, y0 - x, val)
        set_array_val(array, x0 - y, y0 - x, val)
    return array


def zero_value_on_circle(array, x0, y0, r):
    """Sums all values that lie on a circle with radius r around x0, y0 on an array. Algorithm: Midpoint circle algorithm."""
    f = 1 - r
    ddf_x = 1
    ddf_y = -2 * r
    x = 0
    y = r
    if get_array_val(array, x0, y0 + r, 1) == 0.0:
        return True
    if get_array_val(array, x0, y0 - r, 1) == 0.0:
        return True
    if get_array_val(array, x0 + r, y0, 1) == 0.0:
        return True
    if get_array_val(array, x0 - r, y0, 1) == 0.0:
        return True

    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        if get_array_val(array, x0 + x, y0 + y, 1) == 0.0:
            return True
        if get_array_val(array, x0 - x, y0 + y, 1) == 0.0:
            return True
        if get_array_val(array, x0 + x, y0 - y, 1) == 0.0:
            return True
        if get_array_val(array, x0 - x, y0 - y, 1) == 0.0:
            return True
        if get_array_val(array, x0 + y, y0 + x, 1) == 0.0:
            return True
        if get_array_val(array, x0 - y, y0 + x, 1) == 0.0:
            return True
        if get_array_val(array, x0 + y, y0 - x, 1) == 0.0:
            return True
        if get_array_val(array, x0 - y, y0 - x, 1) == 0.0:
            return True
    return False


def get_colors(plot_lib="matplotlib"):
    colors = []
    color_names = ['lime green', 'cyan', 'orange', 'red', 'magenta', 'warm brown', 'gold', 'grey', 'black',
                   'silver', 'emerald', 'clear blue', 'raspberry', 'electric purple', 'pale pink', 'ocean',
                   'pale green', 'pinkish orange', 'sandy yellow', 'terracotta', 'slime green', 'stormy blue',
                   'vomit', 'topaz', 'flat blue', 'seaweed', 'medium purple', 'dark fuchsia']
    for name in color_names:
        colors.append(matlib_col.to_rgba(matlib_col.get_named_colors_mapping()['xkcd:{}'.format(name)]))
    if plot_lib == "matplotlib":
        return colors
    elif plot_lib == "plotly":
        return [plotly_col.label_rgb(np.array(color[0:3]) * 255) for color in colors]
    else:
        assert False, "Unknown plot_lib"



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
        (0, (1, 10))]  # loosely dotted
    return linestyles


def get_markers():
    return ['.', '1', '2', '3', '4', '+', 'x', '*']


def list_2_str(list):
    string = ""
    for l in list:
        string += f"_{round(l, 3)}"
    return string


def nice_str(str):
    if 'static_risk_x_mirr' in str:
        return 'Risk value (x mirrored)'
    elif 'static_risk_y_mirr' in str:
        return 'Risk value (y mirrored)'
    elif 'static_risk' in str:
        return r"Risk value $f_\mathrm{R}$"
    elif str == 'static_noise_violation_x_mirr':
        return 'Noise immission (x mirrored)'
    elif str == 'static_noise_violation_y_mirr':
        return 'Noise immission (y mirrored)'
    elif str == 'static_noise_violation':
        return r"Noise immission $f_\mathrm{N}$"
    elif str == 'length_network':
        return 'Length in m'
    elif 'static_nfa_violation' in str:
        return 'Violation of No-fly-areas'
    elif str == 'radio_coverage':
        return r"Radio dist. $f_\mathrm{D}$"
    elif str == 'overall_travel_time':
        return 'Travel time in s'
    elif str == 'energy':
        return r"Energy $f_\mathrm{E}$"
    elif str == 'Calc time':
        return 'Calculation time (with overhead) in s'
    elif str == 'Raw calc time':
        return 'Calculation time in s'
    elif str == 'Eval calc time':
        return "Evaluation calculation time (s)"
    elif str == 'Prep. Calc time':
        return "Pre-processing calculation time (s)"
    elif str == 'Convergence':
        return 'Convergence'
    elif str == 'Diversity':
        return 'Diversity'
    elif str == 'GD':
        return 'GD'
    elif str == 'IGD':
        return 'IGD'
    elif str == 'Hyp. Vol.':
        return 'Hypervolume'
    elif str == 'Norm. Calc time':
        return 'Normalized calculation time'
    elif str == 'Norm. Hyp. Vol.':
        return r'Normalized hypervolume $m_\mathrm{NHV}$'
    elif str == 'Func_eval':
        return 'Function evaluations'
    elif str == 'discomfort':
        return 'Discomfort'
    elif str == "CMA-ES_with_AI_True":
        return "H. MO-CMA-ES"
    elif str == "CMA-ES_with_AI_False":
        return "MO-CMA-ES"
    elif str == "NSGA2_with_AI_True":
        return "H. NSGA2"
    elif str == "NSGA2_with_AI_False":
        return "NSGA2"
    elif str == "L-BFGS-B_with_AI_True":
        return "H. L-BFGS-B"
    elif str == "L-BFGS-B_with_AI_False":
        return "L-BFGS-B"
    elif str == "ES_with_AI_True":
        return "H. ES"
    elif str == "ES_with_AI_False":
        return "ES"
    elif "para_dijkstra_res" in str:
        return get_cell_size_from_resolution_param(str.split("_")[-1])
    elif str == "NSGA3_with_AI_True_AdapNumCP_True_Niching_True":
        return "H. NSGA3"
    elif str == "NSGA3_with_AI_False_AdapNumCP_False_Niching_False":
        return "NSGA3"
    elif str == "RVEA_with_AI_True_AdapNumCP_True_Niching_True":
        return "H. RVEA"
    elif str == "RVEA_with_AI_False_AdapNumCP_False_Niching_False":
        return "RVEA"
    elif str == "SMS-EMOA_with_AI_True_AdapNumCP_True_Niching_True":
        return "H. SMS-EMOA"
    elif str == "SMS-EMOA_with_AI_False_AdapNumCP_False_Niching_False":
        return "SMS-EMOA"
    elif str == "NSGA3_with_AI_True_Niching_True":
        return "H. NSGA3"
    elif str == "NSGA3_with_AI_False_Niching_False":
        return "NSGA3"
    elif str == "RVEA_with_AI_True_Niching_True":
        return "H. RVEA"
    elif str == "RVEA_with_AI_False_Niching_False":
        return "RVEA"
    elif str == "SMS-EMOA_with_AI_True_Niching_True":
        return "H. SMS-EMOA"
    elif str == "SMS-EMOA_with_AI_False_Niching_False":
        return "SMS-EMOA"
    elif "NSGA3_with_NumCP" in str:
        num = int(str.split("NSGA3_with_NumCP_")[1])
        return f"{num} CPs"
    elif str == "NSGA3_with_Niching_False":
        return "Without Niching"
    elif str == "NSGA3_with_Niching_True":
        return "With Niching"
    elif str == "HES":
        return "H. ES"
    elif str == "NS":
        return "NSGA2"
    elif str == "HNS":
        return "H. NSGA2"
    elif str == "LB":
        return "L-BFGS-B"
    elif str == "ADIJ":
        return "ADS"
    elif str == "DIJ":
        return "DA"
    elif str == "NSGA3_with_WSP_1":
        return r"$n_{\mathrm{WS}}=4$"
    elif str == "NSGA3_with_WSP_2":
        return r"$n_{\mathrm{WS}}=10$"
    elif str == "NSGA3_with_WSP_3":
        return r"$n_{\mathrm{WS}}=20$"
    elif str == "NSGA3_with_WSP_4":
        return r"$n_{\mathrm{WS}}=35$"
    elif str == "NSGA3_with_WSP_5":
        return r"$n_{\mathrm{WS}}=56$"
    elif str == "NSGA3_with_WSP_6":
        return r"$n_{\mathrm{WS}}=84$"
    elif str == "NSGA3_with_Norm_NONE":
        return "No Normalization"
    elif str == "NSGA3_with_Norm_DBS":
        return "DBS Normalization"
    elif str == "NSGA3_with_Norm_LSTSQ":
        return "Least Square Normalization"
    else:
        return str
        # assert 1==0, 'Nice string of {} not defined.'.format(str)


def get_cell_size_from_resolution_param(res_param):
    """See get_cell_size_from_resolution_param() in tests.get_dijkstra_resolution_test"""
    d = {0: '2280m x 1500m', 2: '1140m x 1500m', 3: '760m x 750m', 4: '570m x 500m', 5: '456m x 500m', 6: '380m x 375m',
         8: '285m x 300m', 10: '228m x 250m', 12: '190m x 250m', 14: '152m x 150m', 18: '120m x 125m',
         20: '114m x 125m', 23: '95m x 100m', 28: '76m x 75m', 35: '60m x 60m', 40: '57m x 60m', 49: '40m x 50m',
         59: '38m x 50m', 69: '30m x 30m', 86: '24m x 25m', 105: '20m x 20m', 117: '19m x 20m', 136: '15m x 15m',
         172: '12m x 12m', 210: '10m x 10m', 257: '8m x 10m', 333: '6m x 6m', 418: '5m x 5m', 513: '4m x 4m',
         666: '3m x 3m', 950: '2m x 2m'}
    return d.get(int(res_param), "xxx")


def fit_number_of_cp(cp_vec, number_of_cp):
    """Gets a vector of numbers and duplicates starting at index 1 the entries with increasing indexing until the length
    of the vector equals number_of_cp."""
    start = 0
    increment = 2
    while len(cp_vec) < number_of_cp:
        if start >= len(cp_vec):
            start = 0
            increment += 1
        cp_vec.insert(start, cp_vec[start])
        start += increment
    return cp_vec


def remove_duplicate_cp(cp_vec, weight_vec=None):
    """Removes duplicate elements in vector and the element with the same index in the second vector, if the elements
    are neighbors."""
    cp_vec = cp.deepcopy(cp_vec)
    weight_vec = cp.deepcopy(weight_vec)
    idx = 0
    while idx < len(cp_vec) - 1:
        if list(cp_vec[idx]) == list(cp_vec[idx + 1]):
            del cp_vec[idx + 1]
            if weight_vec is not None:
                del weight_vec[idx + 1]
            continue
        idx += 1

    return cp_vec, weight_vec


def order_points(points, ind):
    """Orders points so that the nearest are always next to each other in the list"""
    points = cp.deepcopy(points)
    points_new = [points.pop(ind)]  # initialize a new list of points with the known first point
    pcurr = points_new[-1]  # initialize the current point (as the known point)
    while len(points) > 0:
        d = np.linalg.norm(np.array(points) - np.array(pcurr),
                           axis=1)  # distances between pcurr and all other remaining points
        ind = d.argmin()  # index of the closest point
        points_new.append(points.pop(ind))  # append the closest point to points_new
        pcurr = points_new[-1]  # update the current point
    return points_new


def chamfer_distance(x, y, metric='euclidean', direction='bi', algo="kd_tree"):
    """
    Taken from: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            r"sum_{x_i in x}{min_{y_j \n y}{||x_i-y_j||**2}} + sum_{y_j in y}{min_{x_i in x}{||x_i-y_j||**2}}"
    """
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    elif direction == 'min':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = min(np.sum(min_y_to_x), np.sum(min_x_to_y))
    elif direction == 'max':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = max(np.sum(min_y_to_x), np.sum(min_x_to_y))
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False,
                                clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
        Code from: https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) examples if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom;
    t1 = detB / denom;

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def get_time():
    a = dt.now()
    return a.strftime('%y %m %d %H:%M:%S')
