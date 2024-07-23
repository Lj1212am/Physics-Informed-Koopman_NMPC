from scipy.spatial.transform import Rotation
import numpy as np


class SE3Control(object):
    def __init__(self):

        # Quadrotor physical parameters (may be different from true system parameters in crazyflie_params.py)
        self.mass               = 0.03
        self.Ixx                = 1.43e-5
        self.Iyy                = 1.43e-5
        self.Izz                = 2.89e-5

        self.arm_length         = 0.046
        self.rotor_speed_min    = 0
        self.rotor_speed_max    = 2500
        self.k_thrust           = 2.3e-08
        self.k_drag             = 7.8e-11

        self.inertia            = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.g                  = 9.81

        self.pos_kp             = 10
        self.pos_kd             = 2 * 1.0 * np.sqrt(self.pos_kp)
        self.posz_kp            = 10
        self.posz_kd            = 2 * 1.0 * np.sqrt(self.pos_kp)
        self.geo_rollpitch_kp   = 500
        self.geo_rollpitch_kd   = 2 * 0.5 * np.sqrt(self.geo_rollpitch_kp)
        self.geo_yaw_kp         = 5
        self.geo_yaw_kd         = 2 * 1.15 * np.sqrt(self.geo_yaw_kp)

        self.pos_kp_mat = np.diag(np.array([self.pos_kp, self.pos_kp, self.posz_kp]))
        self.pos_kd_mat = np.diag(np.array([self.pos_kd, self.pos_kd, self.posz_kd]))
        self.att_kp_mat = np.diag(np.array([self.geo_rollpitch_kp, self.geo_rollpitch_kp, self.geo_yaw_kp]))
        self.att_kd_mat = np.diag(np.array([self.geo_rollpitch_kd, self.geo_rollpitch_kd, self.geo_yaw_kd]))

        k                       = self.k_drag / self.k_thrust
        self.ctrl_forces_map    = np.array([[1, 1, 1, 1],
                                            [0, self.arm_length, 0, -self.arm_length],
                                            [-self.arm_length, 0, self.arm_length, 0],
                                            [k, -k, k, -k]])
        self.forces_ctrl_map    = np.linalg.inv(self.ctrl_forces_map)
        self.trim_motor_spd     = 1790.0
        trim_force              = self.k_thrust * np.square(self.trim_motor_spd)
        self.forces_old         = np.array([trim_force, trim_force, trim_force, trim_force])

    def update(self, t, state, flat_output):
        # print('state', state)
        # print('flat output', flat_output)
        pos         = state[0:3]
        vel         = state[3:6]
        quats       = state[6:10]
        rates       = state[10:13]
        pos_des     = flat_output[0:3]
        vel_des     = flat_output[3:6]
        yaw_des     = flat_output[15]

        print('diff translational ', state[:6] - flat_output[:6])

        # Get rotation matrix, from quaternions
        r           = Rotation.from_quat(quats)
        rot_mat     = r.as_matrix()

        t11         = rot_mat[0, 0]
        t12         = rot_mat[0, 1]
        t13         = rot_mat[0, 2]
        t21         = rot_mat[1, 0]
        t22         = rot_mat[1, 1]
        t23         = rot_mat[1, 2]
        t31         = rot_mat[2, 0]
        t32         = rot_mat[2, 1]
        t33         = rot_mat[2, 2]

        # Get Euler angles from rotation matrix
        phi         = np.arcsin(t32)
        tet         = np.arctan2(-t31 / np.cos(phi), t33 / np.cos(phi))
        psi         = np.arctan2(-t12 / np.cos(phi), t22 / np.cos(phi))

        # Position controller
        r_ddot_des  = -(self.pos_kd_mat @ (vel - vel_des)) - (self.pos_kp_mat @ (pos - pos_des))

        # Geometric nonlinear controller
        f_des       = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g])
        f_des       = np.squeeze(f_des)
        b3          = rot_mat @ np.array([0, 0, 1])
        b3_des      = f_des / np.linalg.norm(f_des)
        a_psi       = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des      = np.cross(b3_des, a_psi) / np.linalg.norm(np.cross(b3_des, a_psi))
        rot_des     = np.array([[np.cross(b2_des, b3_des)], [b2_des], [b3_des]]).T
        rot_des     = np.squeeze(rot_des)
        err_mat     = 0.5 * (rot_des.T @ rot_mat - rot_mat.T @ rot_des)
        err_vec     = np.array([-err_mat[1, 2], err_mat[0, 2], -err_mat[0, 1]])

        u1          = np.array([b3 @ f_des])
        u2          = self.inertia @ (-self.att_kp_mat @ err_vec - self.att_kd_mat @ rates)
        # print('u1', u1)
        # print('u2', u2)

        # Compute motor speed commands
        forces = self.forces_ctrl_map @ np.concatenate((u1, u2))
        # print('forces', forces)
        # Protect against invalid force and motor speed commands (set them to previous motor speeds)
        forces[forces < 0]  = np.square(self.forces_old[forces < 0]) * self.k_thrust
        cmd_motor_speeds    = np.sqrt(forces / self.k_thrust)
        self.forces_old     = forces

        # Software limits for motor speeds
        cmd_motor_speeds    = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # print('cmd motor speed', cmd_motor_speeds)
        # Not used in simulation, for analysis only
        forces_limited  = self.k_thrust * np.square(cmd_motor_speeds)
        ctrl_limited    = self.ctrl_forces_map @ forces_limited
        cmd_thrust      = ctrl_limited[0]
        cmd_moment      = ctrl_limited[1:]
        r               = Rotation.from_matrix(rot_des)
        cmd_q           = r.as_quat()

        # print('hstack', np.hstack(cmd_motor_speeds))
        # print(np.array_equal(np.hstack(cmd_motor_speeds), cmd_motor_speeds))
        return np.hstack((cmd_motor_speeds, cmd_thrust, cmd_moment, cmd_q, r_ddot_des))
