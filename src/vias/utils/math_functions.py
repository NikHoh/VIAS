import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-10


def inverse_square_law_with_offset(x, y, z, x0, y0, z0, P):
    # Calculate the Euclidean distance r from the point (x0, y0, z0)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    # Apply the inverse square law formula
    # offset +1 applied so that value is P at source point (x0, y0, z0)
    return P / ((r + 1) ** 2)


def chamfer_distance(x, y, metric="euclidean", direction="bi", algo="kd_tree"):
    """
    Taken and adapted from:
    https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or
        scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            r"sum_{x_i in x}{min_{y_j \n y}{||x_i-y_j||**2}} + sum_{y_j in y}{min_{
            x_i in x}{||x_i-y_j||**2}}"
    """
    chamfer_dist = np.inf
    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction in ["bi", "min", "max"]:
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm=algo, metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        if direction == "bi":
            chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        elif direction == "min":
            chamfer_dist = min(np.sum(min_y_to_x), np.sum(min_x_to_y))
        elif direction == "max":
            chamfer_dist = max(np.sum(min_y_to_x), np.sum(min_x_to_y))
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")
    return chamfer_dist
