from casadi import * #Function, printme, MX, vertcat, mtimes, sum1, horzcat, if_else
import numpy as np
import math


# def rk4(ode, h, x, u):
#     k1 = ode(x, u)
#     k2 = ode(x + h / 2 * k1, u)
#     k3 = ode(x + h / 2 * k2, u)
#     k4 = ode(x + h * k3, u)
#     xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#
#     q_norm = safe_normalize(xf[6:10])
#     xf = vertcat(xf[0:6], q_norm, xf[10:])
#
#     return xf

# def rk4(ode, h, x, u):
#     k1 = ode(x, u)
#     k2 = ode(x + (h / 2.0) * k1, u)
#     k3 = ode(x + (h / 2.0) * k2, u)
#     k4 = ode(x + h * k3, u)
#     xf = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#
#     # # q_norm = safe_normalize(xf[6:10])
#     # xf = vertcat(xf[0:6], q_norm, xf[10:])
#
#     return xf

def vee_map(matrix):
    """
    Convert a skew-symmetric matrix to a vector using CasADi.

    Args:
    - matrix: A skew-symmetric matrix represented as a CasADi MX type.

    Returns:
    - A CasADi MX vector representing the angular velocity.
    """
    # Extract the angular velocity vector components from the skew-symmetric matrix
    vector = MX([-matrix[1, 2], matrix[0, 2], -matrix[0, 1]])
    return vector


from casadi import MX, norm_2, sin, cos


def rotation_matrix_to_quaternion(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    # Using max to avoid taking sqrt of negative numbers
    S = sqrt(fmax(tr + 1.0, 0)) * 2
    qw = 0.25 * S
    qx = (m21 - m12) / S
    qy = (m02 - m20) / S
    qz = (m10 - m01) / S

    # Direct construction of the quaternion SX vector
    q = vertcat(qx, qy, qz, qw)

    return q

def axis_angle_to_quaternion(v):
    """
    Convert a rotation vector (axis-angle representation) to a quaternion.

    v: A 3x1 CasADi MX vector representing the rotation vector.

    Returns a quaternion [x, y, z, w] as a CasADi MX vector.
    """
    theta = norm_2(v)  # Magnitude of the rotation vector
    n = v / theta  # Normalized rotation axis

    w = cos(theta / 2)
    x = n[0] * sin(theta / 2)
    y = n[1] * sin(theta / 2)
    z = n[2] * sin(theta / 2)

    quaternion = MX([x, y, z, w])
    return quaternion


def quaternion_to_rotation_matrix_casadi():
    # Define a quaternion as a CasADi MX type variable
    quat = MX.sym('quat', 4)
    #
    # norm_q = sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
    # normalized_quat = quat / norm_q

    # Extract quaternion elements
    # x, y, z, w = normalized_quat[0], normalized_quat[1], normalized_quat[2], normalized_quat[3]
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Calculate the rotation matrix elements
    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)

    # Constructing the rotation matrix with calculated elements
    rot_mat = MX(3, 3)
    rot_mat[0, 0] = r00
    rot_mat[0, 1] = r01
    rot_mat[0, 2] = r02
    rot_mat[1, 0] = r10
    rot_mat[1, 1] = r11
    rot_mat[1, 2] = r12
    rot_mat[2, 0] = r20
    rot_mat[2, 1] = r21
    rot_mat[2, 2] = r22

    # Create the CasADi function
    quaternion_to_rot_matrix = Function('quaternion_to_rot_matrix', [quat], [rot_mat])

    return quaternion_to_rot_matrix

def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a rotation matrix.
    quat: Quaternion vector [x, y, z, w] represented as a CasADi MX type,
          where w is the scalar part.
    """
    # Reordering quat to correctly place the scalar part as the fourth element


    # quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    # # Calculate the norm of the quaternion
    # norm_q = sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
    #
    # # Normalize the quaternion
    # normalized_q = quat / norm_q
    #
    # x, y, z, w = normalized_q[0], normalized_q[1], normalized_q[2], normalized_q[3]
    # print('x, y, z, w:', x, y, z, w)

    # Rotation matrix components using the corrected quaternion order [x, y, z, w]
    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)
    #
    # # Create and return the rotation matrix
    # rot_matrix = np.array([[r00, r01, r02],
    #                        [r10, r11, r12],
    #                        [r20, r21, r22]])


    # Constructing the rotation matrix with corrected formulae
    rot_mat = MX.zeros(3, 3)
    rot_mat[0, 0] = r00
    rot_mat[0, 1] = r01
    rot_mat[0, 2] = r02
    rot_mat[1, 0] = r10
    rot_mat[1, 1] = r11
    rot_mat[1, 2] = r12
    rot_mat[2, 0] = r20
    rot_mat[2, 1] = r21
    rot_mat[2, 2] = r22


    return rot_mat

def rk4(ode_func, h, state_dim, control_dim):
    # Define symbolic variables for state (x) and control input (u)
    x = MX.sym('x', state_dim)  # state_dim should be defined according to your state vector size
    u = MX.sym('u', control_dim)  # control_dim should be defined as per your control vector size

    # RK4 steps using the provided ode function
    k1 = ode_func(x, u)
    k2 = ode_func(x + (h / 2.0) * k1, u)
    k3 = ode_func(x + (h / 2.0) * k2, u)
    k4 = ode_func(x + h * k3, u)
    xf = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Normalize the quaternion part of the state vector
    # q_norm = safe_normalize(xf[6:10])
    # xf = vertcat(xf[0:6], q_norm, xf[10:])

    # Create and return a Casadi Function for the RK4 step
    rk4_step = Function('rk4_step', [x, u], [xf]) #, ['x', 'u'], ['xf'])
    return rk4_step


def quaternion_to_yaw(quat):
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    # Normalize the quaternion
    norm = math.sqrt(x ** 2 + y ** 2 + z ** 2 + w ** 2)
    x_n = x / norm
    y_n = y / norm
    z_n = z / norm
    w_n = w / norm

    # Calculate yaw
    yaw = math.atan2(2.0 * (w_n * z_n + x_n * y_n), 1.0 - 2.0 * (y_n ** 2 + z_n ** 2))
    return yaw


# def rk4(ode_func, h, state_dim, control_dim):
#     # Define symbolic variables for state (x) and control input (u)
#     x = MX.sym('x', state_dim)
#     u = MX.sym('u', control_dim)
#
#     # RK4 steps using the provided ode function
#     k1 = ode_func(x, u)
#     # k1_print = printme('k1: ', k1)  # Print k1 when it's evaluated
#     k2 = ode_func(x + (h / 2.0) * k1, u)
#     # k2_print = printme('k2: ', k2)  # Print k2 when it's evaluated
#     k3 = ode_func(x + (h / 2.0) * k2, u)
#     k4 = ode_func(x + h * k3, u)
#     xf = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#     # monitor()
#
#     # Create and return a Casadi Function for the RK4 step
#     rk4_step = Function('rk4_step', [x, u], [xf, k1, k2, k3, k4], ['x', 'u'], ['xf', 'k1', 'k2', 'k3', 'k4'])
#     return rk4_step


# def rotate_k(q):
#     k_vector = vertcat(
#         2.0 * (q[0] * q[2] + q[1] * q[3]),
#         2.0 * (q[1] * q[2] - q[0] * q[3]),
#         (1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2))
#     )
#     return k_vector
#
#
# def hat_map(s):
#     return vertcat(
#         horzcat(0, -s[2], s[1]),
#         horzcat(s[2], 0, -s[0]),
#         horzcat(-s[1], s[0], 0)
#     )
#
#
# def quat_dot(quat, omega):
#     q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
#
#     # Define the G matrix as per the provided function
#     G = vertcat(
#         horzcat(q3, q2, -q1, -q0),
#         horzcat(-q2, q3, q0, -q1),
#         horzcat(q1, -q0, q3, -q2)
#     )
#
#     # Calculate the quaternion derivative
#     quat_dot = 0.5 * mtimes(G.T, omega)
#
#     # Calculate the error in the quaternion (to maintain normalization)
#     quat_err = sum1(quat ** 2) - 1
#     quat_err_grad = 2 * quat
#
#     # Correct the quaternion derivative to account for the error
#     quat_dot -= quat_err * quat_err_grad
#
#     return quat_dot

def rotate_k():
    q_sym = MX.sym('q', 4)
    k_vector = vertcat(
        2.0 * (q_sym[0] * q_sym[2] + q_sym[1] * q_sym[3]),
        2.0 * (q_sym[1] * q_sym[2] - q_sym[0] * q_sym[3]),
        (1.0 - 2.0 * (q_sym[0] ** 2.0 + q_sym[1] ** 2.0))
    )
    return Function('rotate_k', [q_sym], [k_vector])


def hat_map():

    s = MX.sym('s', 3)
    hat_map = vertcat(
        horzcat(0.0, -s[2], s[1]),
        horzcat(s[2], 0.0, -s[0]),
        horzcat(-s[1], s[0], 0.0)
    )

    return Function('hat_map', [s], [hat_map])

def quat_dot():
    quat = MX.sym('q', 4)
    omega = MX.sym('omega', 3)
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]

    # Define the G matrix as per the provided function
    G = vertcat(
        horzcat(q3, q2, -q1, -q0),
        horzcat(-q2, q3, q0, -q1),
        horzcat(q1, -q0, q3, -q2)
    )

    # Calculate the quaternion derivative
    quat_dot = 0.5 * mtimes(G.T, omega)

    # Calculate the error in the quaternion (to maintain normalization)
    quat_err = sum1(quat ** 2) - 1
    quat_err_grad = 2 * quat

    # Correct the quaternion derivative to account for the error
    quat_dot -= quat_err * quat_err_grad

    return Function('quat_dot', [quat, omega], [quat_dot])


# def quat_dot(quat, omega):
#     # Ensure the quaternion is normalized
#     quat = quat / np.linalg.norm(quat)
#
#     # Quaternion multiplication
#     omega_quat = np.hstack(([0.0], omega))
#     quat_dot = 0.5 * quaternion_multiply(quat, omega_quat)
#     return quat_dot
#
#
# def quaternion_multiply(q1, q2):
#     # Quaternion multiplication q1 * q2
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return np.array([w, x, y, z])

# def quat_multiply(q1, q2):
#     """
#     Quaternion multiplication with components ordered as [x, y, z, w].
#     """
#     x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
#     x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
#
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
#
#     return vertcat(x, y, z, w)

# def quat_dot(quat, omega_vec):
#     """
#     Computes the time derivative of the quaternion given the angular velocity.
#     The quaternion is represented as [x, y, z, w].
#     """
#     x, y, z, w = quat[0], quat[1], quat[2], quat[3]
#     # Convert angular velocity vector to quaternion with zero scalar part
#     omega_quat = vertcat(0, omega_vec[0], omega_vec[1], omega_vec[2])
#
#     # Adjust for the quaternion order [x, y, z, w] in multiplication
#     q_dot = 0.5 * quat_multiply(vertcat(x, y, z, w), omega_quat)
#
#     return q_dot

def safe_normalize(q, eps=1e-3):
    norm_q = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q_normalized = q / norm_q
    # # Avoid division by zero by adding a small epsilon to the norm

    q_normalized = if_else(norm_q > eps, q_normalized, q)
    # norm = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    # print(q / norm)
    return q_normalized

