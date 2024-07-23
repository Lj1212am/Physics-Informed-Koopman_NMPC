from koopman.imports import *
from koopman.matrix_math import hat_map
from scipy.spatial.transform import Rotation as R


def quad_dynamics(t, x, u, params):
    """
    Calculates the dynamics of a quadrotor.

    Parameters:
    x (numpy.ndarray): The state vector of the quadrotor.
    u (numpy.ndarray): The control input vector.
    params (dict): A dictionary of parameters (mass, J, g).

    Returns:
    numpy.ndarray: The derivative of the state vector.
    """
    # Extracting parameters

    # print('thrust', u[0])

    mass = params['mass']
    J = params['J']
    g = params['g']
    e3 = np.array([0, 0, 1])

    # Extracting and reshaping states
    v = x[3:6]#.reshape((3, 1))
    R_mat = x[6:15].reshape((3, 3))
    omega = x[15:18]#.reshape((3, 1))

    # Quaternion from rotation matrix
    # q = R.from_matrix(R_mat.transpose()).as_quat()

    q = x[18:22]

    # Control inputs
    F = u[0]
    # F = u[0] * rotate_k(q)
    M = u[1:4]#.reshape((3, 1))
    # print('M', M)

    # Dynamics
    a = R_mat @ (1 / mass * F * e3) - g * e3
    # a = F / mass + np.array([0, 0, -g])

    # print('omega', omega)

    dR = R_mat @ (hat_map(omega))

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    G = np.array([[q3, q2, -q1, -q0],
                  [-q2, q3, q0, -q1],
                  [q1, -q0, q3, -q2]])
    dq = 0.5 * G.T @ omega

    # dq = 0.5 * G.dot(omega)
    quat_err = np.sum(q**2) - 1
    quat_err_grad = 2 * q
    dq = dq - quat_err * quat_err_grad


    omega_hat = hat_map(omega)

    omega_dot = np.linalg.inv(J).dot(M - omega_hat.dot(J.dot(omega)))

    # print('omega_dot', omega_dot)
    # print('omega', omega.shape)
    # omega_dot = np.linalg.inv(J) @ (M - omega_hat @ (J @ omega.reshape((3, 1))))
    # print('omega dot', omega_dot)
    # States_dot
    x_dot = np.concatenate((v, a.flatten(), dR.flatten(), omega_dot, dq))

    return x_dot

def hat_map(omega):
    """Calculate the hat map of a vector for cross product operations."""
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])

def rotate_k(q):
    return np.array([2*(q[0]*q[2]+q[1]*q[3]),
                       2*(q[1]*q[2]-q[0]*q[3]),
                     1-2*(q[0]**2  +q[1]**2)])