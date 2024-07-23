from koopman.imports import *


def EDMD(Z1, Z2, U):
    Z1_aug = np.vstack((Z1, U))

    m = Z1.shape[1]
    Y = np.dot(Z2, Z1_aug.T) / m
    G = np.dot(Z1_aug, Z1_aug.T) / m

    K = np.dot(Y, np.linalg.pinv(G))
    A = K[:, :Z1.shape[0]]
    B = K[:, Z1.shape[0]:]

    return A, B