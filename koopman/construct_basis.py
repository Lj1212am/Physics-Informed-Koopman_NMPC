from koopman.imports import *
from koopman.matrix_math import hat_map


def construct_basis(X, n_basis):
    # Extract states
    R = X[6:15].reshape(3, 3, order='F')
    wb = X[15:18].reshape(3, 1)
    wb = wb.squeeze()
    Q = X[18:22].reshape(4, 1)
    wb_hat = hat_map(wb)

    # Build observables
    basis = np.zeros((9 * n_basis, 1))
    Z = R
    for i in range(n_basis):
        Z = np.dot(Z, wb_hat)
        basis[i * 9:(i + 1) * 9, :] = Z.flatten(order='F').reshape(9, 1)

    # print(wb.reshape(-1, 1).shape)
    # Concatenate results
    basis_result = np.concatenate([R.flatten(order='F').reshape(9, 1), wb.reshape(-1, 1), Q, basis])

    return basis_result
