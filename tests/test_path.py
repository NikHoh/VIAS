import numpy as np

from vias.path import Path


def get_test_path() -> Path:
    vec_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
    vec_y = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]).astype(float)
    vec_z = np.zeros(vec_x.shape)
    p = Path([vec_x, vec_y, vec_z])
    return p


def main():
    p = get_test_path()
    p.approximate_nurbs(5, 3, 2)


if __name__ == '__main__':
    main()
