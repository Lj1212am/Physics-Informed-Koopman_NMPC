from koopman.imports import *
from koopman.matrix_math import vee_map

def compute_rmse(X, X_ref):
    x_ref = X_ref[0:3, :]
    dx_ref = X_ref[3:6, :]
    wb_ref = X_ref[15:18, :]
    x = X[0:3, :]
    dx = X[3:6, :]
    wb = X[15:18, :]

    theta_err = np.zeros((3, X.shape[1]))
    angle_err = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        q_true = Quaternion(X_ref[18:22, i])
        q1 = Quaternion(X[18:22, i])

        R_true = q_true.rotation_matrix
        R = q1.rotation_matrix

        theta_err[:, i] = 0.5 * vee_map(R_true.T @ R - R.T @ R_true)

        q_est = q1.conjugate.normalised
        q_temp = q_est * q_true

        angle_err[:, i] = 2 * np.arctan2(np.linalg.norm([q_temp.x, q_temp.y, q_temp.z]), q_temp.w)

    x_error = x_ref.flatten() - x.flatten()
    RMSE_x = np.sqrt(np.mean(x_error**2))

    dx_error = dx_ref.flatten() - dx.flatten()
    RMSE_dx = np.sqrt(np.mean(dx_error**2))

    ang_error = angle_err
    RMSE_quat_angle = np.sqrt(np.mean(ang_error**2))

    theta_error = theta_err.flatten()
    RMSE_theta = np.sqrt(np.mean(theta_error**2))

    wb_error = wb_ref.flatten() - wb.flatten()
    RMSE_wb = np.sqrt(np.mean(wb_error**2))

    return RMSE_x, RMSE_dx, RMSE_quat_angle, RMSE_theta, RMSE_wb
