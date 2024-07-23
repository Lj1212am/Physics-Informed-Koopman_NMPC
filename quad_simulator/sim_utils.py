import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import scipy


def data_plotting(time_sim, state, state_dot, ref, control):
    x               = state[:, 0:3]
    v               = state[:, 3:6]
    q               = state[:, 6:10]
    w               = state[:, 10:13]
    x_des           = ref[:, 0:3]
    v_des           = ref[:, 3:6]
    # s               = control[:, 0:4]
    T               = control[:, 0]
    M               = control[:, 1:4]
    # cmd_quats       = control[:, 8:12]
    # accel_des       = control[:, 12:15]
    # print('q', q)
    q1              = np.squeeze(q)
    # print('q_squeezed', q1)
    r               = Rotation.from_quat(q1)
    euler_out       = r.as_euler('zyx')
    psi             = euler_out[:, 0]
    # print('psi', psi)
    # print('yaw', np.degrees(psi))
    tet             = euler_out[:, 1]
    phi             = euler_out[:, 2]

    rot_mat         = r.as_matrix()
    vel_b           = np.zeros_like(v)
    for i in range(rot_mat.shape[0]):
        vel_b[i, :] = rot_mat[i, :, :].T @ v[i, :]

    # q1              = np.squeeze(cmd_quats)
    # r               = Rotation.from_quat(q1)
    # euler_out       = r.as_euler('zyx')
    # psi_des         = euler_out[:, 0]
    # tet_des         = euler_out[:, 1]
    # phi_des         = euler_out[:, 2]

    # Figure 1 - 3D path
    N       = time_sim.shape[0]-1
    fig     = plt.figure('3D Path')
    ax      = plt.axes(projection='3d')
    ax.plot([x[0, 0]], [x[0, 1]], [x[0, 2]], 'g.', markersize=12, markeredgewidth=2, markerfacecolor='none')
    ax.plot([x[N-1, 0]], [x[N-1, 1]], [x[N-1, 2]], 'r.', markersize=12, markeredgewidth=2, markerfacecolor='none')
    ax.plot(x[:, 0], x[:, 1], x[:, 2], 'b')
    ax.plot(x_des[:, 0], x_des[:, 1], x_des[:, 2], 'k')
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.set_zlim([-3.5, 3.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Figure 2 - Time histories of states
    (fig, axes) = plt.subplots(nrows=4, ncols=2, sharex=True, num='States')
    # Position
    ax = axes[0, 0]
    ax.plot(time_sim, x[:, 0], 'b',    time_sim, x[:, 1], 'r',    time_sim, x[:, 2], 'g')
    ax.plot(time_sim[:], x_des[:, 0], 'b--', time_sim[:], x_des[:, 1], 'r--', time_sim[:], x_des[:, 2], 'g--')
    ax.legend(('x', 'y', 'z'))
    ax.set_ylabel('Pos, m')
    ax.grid('major')
    # Velocity
    ax = axes[1, 0]
    ax.plot(time_sim, v[:, 0], 'b',    time_sim, v[:, 1], 'r',    time_sim, v[:, 2], 'g')
    ax.plot(time_sim[:], v_des[:, 0], 'b--', time_sim[:], v_des[:, 1], 'r--', time_sim[:], v_des[:, 2], 'g--')
    ax.set_ylabel('Vel, m/s')
    ax.set_xlabel('time, s')
    ax.grid('major')
    # Orientation
    ax = axes[0, 1]
    # ax.plot(time_sim, np.degrees(phi_des), 'b--', time_sim, np.degrees(tet_des), 'r--', time_sim, np.degrees(psi_des), 'g--')
    ax.plot(time_sim, np.degrees(phi), 'b', time_sim, np.degrees(tet), 'r', time_sim, np.degrees(psi), 'g')
    ax.legend(('roll', 'pitch', 'yaw'))
    ax.set_ylabel('Euler angles,deg')
    ax.grid('major')
    # Angular Velocity
    ax = axes[1, 1]
    ax.plot(time_sim, np.degrees(w[:, 0]), 'b', time_sim, np.degrees(w[:, 1]), 'r', time_sim, np.degrees(w[:, 2]), 'g')
    ax.set_ylabel('AngVel,deg/s')
    ax.grid('major')
    # Moment commands
    ax = axes[2, 0]
    ax.plot(time_sim[:], M[:, 0], 'b', time_sim[:], M[:, 1], 'r', time_sim[:], M[:, 2], 'g')
    ax.set_ylabel('Moments, Nm')
    ax.legend(('roll', 'pitch', 'yaw'))
    ax.grid('major')
    ax = axes[2, 1]
    ax.plot(time_sim[:], T, 'b')
    ax.set_ylabel('Thrust, N')
    ax.grid('major')


def get_metrics(state, ref, desired_speed, length):
    print("Commanded speed [m/s]: ", desired_speed)
    print("Trajectory radius [m]: ", length)

    x = state[:, 0:3]
    v = state[:, 3:6]
    q = state[:, 6:10]
    w = state[:, 10:13]
    x_des = ref[:, 0:3]
    v_des = ref[:, 3:6]

    # pos_rmse = np.sqrt((np.square(x_des - x)).mean())
    # vel_rmse = np.sqrt((np.square(v_des - v)).mean())
    pos_rmse = np.sqrt(np.sum(np.square(np.subtract(x, x_des))) / (x.shape[0] * x.shape[1]))
    vel_rmse = np.sqrt(np.sum(np.square(np.subtract(v, v_des))) / (v.shape[0] * v.shape[1]))
    print('\nRMSE:\npos [m]:', pos_rmse, '\nvel [m/s]:', vel_rmse)

    return pos_rmse, vel_rmse


def extract_data(time_sim, state, ref, control, state_nom, state_err):
    x               = state[:, 0:3]
    v               = state[:, 3:6]
    q               = state[:, 6:10]
    w               = state[:, 10:13]
    x_des           = ref[:, 0:3]
    v_des           = ref[:, 3:6]
    # s               = control[:, 0:4]
    T               = control[:, 0]
    M               = control[:, 1:4]
    # cmd_quats       = control[:, 8:12]
    # accel_des       = control[:, 12:15]

    q1                  = np.squeeze(q)
    r                   = Rotation.from_quat(q1)
    rot_mat             = r.as_matrix()
    rot_mat_reshaped    = rot_mat.reshape((q.shape[0], 9))

    # convert xyzw to wxyz
    q_saved = np.zeros((q.shape[0], 4))
    q_saved[:, 0] = q[:, 3]
    q_saved[:, 1] = q[:, 0]
    q_saved[:, 2] = q[:, 1]
    q_saved[:, 3] = q[:, 2]

    ##################### Development code - Test and verify before use! #####################
    # state_nom = ref
    #
    # state_err = state - state_nom


    x_nom         = state_nom[:, 0:3]
    v_nom         = state_nom[:, 3:6]
    q_nom         = state_nom[:, 6:10]
    w_nom         = state_nom[:, 10:13]

    x_err         = state_err[:, 0:3]
    v_err         = state_err[:, 3:6]
    q_err         = state_err[:, 6:10]
    w_err         = state_err[:, 10:13]

    q1_nom                  = np.squeeze(q_nom)
    r_nom                   = Rotation.from_quat(q1_nom)
    rot_mat_nom             = r_nom.as_matrix()
    rot_mat_reshaped_nom    = rot_mat_nom.reshape((q_nom.shape[0], 9))

    # convert xyzw to wxyz
    q_saved_nom  = np.zeros((q_nom.shape[0], 4))
    q_saved_nom[:, 0] = q_nom[:, 3]
    q_saved_nom[:, 1] = q_nom[:, 0]
    q_saved_nom[:, 2] = q_nom[:, 1]
    q_saved_nom[:, 3] = q_nom[:, 2]

    q1_err = np.squeeze(q_err)

    q1_err = np.array(q1_err)  # Replace with your quaternion array
    norms = np.linalg.norm(q1_err, axis=1)

    # Check for zero norm and handle accordingly
    valid_indices = norms > 0
    q1_err_nonzero = q1_err[valid_indices]

    # Normalize non-zero quaternions
    q1_err_normalized = q1_err_nonzero / norms[valid_indices][:, None]

    # Now safely convert to rotations
    r_err = Rotation.from_quat(q1_err_normalized)

    # r_err = Rotation.from_quat(q1_err)
    rot_mat_err = r_err.as_matrix()
    rot_mat_reshaped_err = rot_mat_err.reshape((q_err.shape[0], 9))

    # convert xyzw to wxyz
    q_saved_err = np.zeros((q_err.shape[0], 4))
    q_saved_err[:, 0] = q_err[:, 3]
    q_saved_err[:, 1] = q_err[:, 0]
    q_saved_err[:, 2] = q_err[:, 1]
    q_saved_err[:, 3] = q_err[:, 2]

    # Data collection for koopman testing in MATLAB
    data = np.hstack([np.expand_dims(time_sim, 1), x, v, rot_mat_reshaped, w, q_saved, np.expand_dims(T, 1), M])
    data_nom = np.hstack([np.expand_dims(time_sim, 1), x_nom, v_nom, rot_mat_reshaped_nom, w_nom, q_saved_nom, np.expand_dims(T, 1), M])
    data_err = np.hstack([np.expand_dims(time_sim, 1), x_err, v_err, rot_mat_reshaped_err, w_err, q_saved_err, np.expand_dims(T, 1), M])
    scipy.io.savemat('koopman_dat_err_wxyz.mat', {'data': data, 'data_nom': data_nom, 'data_err': data_err})
    ##################### Development code - Test and verify before use! #####################