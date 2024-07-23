from koopman.imports import *


def hat_map(vector):
    """
    Convert a vector into a skew-symmetric matrix.
    """
    matrix = np.array([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]
                       ], dtype=np.float64)
    return matrix


def vee_map(matrix):
    """
    Convert a skew-symmetric matrix to a vector.
    """
    vector = [-matrix[1][2], matrix[0][2], -matrix[0][1]]
    return np.array(vector)
