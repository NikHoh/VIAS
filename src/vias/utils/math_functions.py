import numpy as np

EPS = 1E-10

def inverse_square_law_with_offset(x, y, z, x0, y0, z0, P):
    # Calculate the Euclidean distance r from the point (x0, y0, z0)
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    # Apply the inverse square law formula
    return P / ((r+1) ** 2)  # offset +1 applid so that value is P at source point (x0, y0, z0)