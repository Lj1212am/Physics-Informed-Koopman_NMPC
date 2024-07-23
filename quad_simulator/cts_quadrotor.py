import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation

# option to include aerodyn drag
include_aerodrag = False


class QuadrotorDynamics(object):
    def __init__(self, quad_params):
        self.mass            = quad_params['mass']  # kg
        self.Ixx             = quad_params['Ixx']   # kg*m^2
        self.Iyy             = quad_params['Iyy']   # kg*m^2
        self.Izz             = quad_params['Izz']   # kg*m^2
        self.arm_length      = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust        = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']    # Nm/(rad/s)**2

        # Additional constants
        self.inertia        = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g              = 9.81  # m/s^2

        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [ k, -k,  k, -k]])
        self.inv_inertia = inv(self.inertia)

        ##################### Development code - Test and verify before use! #####################
        # Parameters for nominal quad model (these are also the ones used in the controller, se3_control.py)
        self.mass_nom = 0.03 * 2
        self.Ixx_nom = 1.43e-5 * 1.5
        self.Iyy_nom = 1.43e-5 * 1.5
        self.Izz_nom = 2.89e-5 * 1.2

        self.inertia_nom = np.diag(np.array([self.Ixx_nom, self.Iyy_nom, self.Izz_nom]))
        self.inv_inertia_nom = inv(self.inertia_nom)
        ##################### Development code - Test and verify before use! #####################

        # Add noise to thrust and moments
        self.cnt = 0
        self.F_d = np.random.normal(size=(15000, 1), scale=1e-3)
        self.M_d = np.random.normal(size=(15000, 3), scale=1e-3)
        # TODO add noise for states
        # TODO seperate out state areas because they could be at different scales
        # num_samples = sim_duration / sampling_rate
        # self.state_noise = np.random.normal('num_samples[0]', '13')
        self.state_noise = None

        self.rotor_spd = np.zeros((4,))

    def step(self, state, cmd_rotor_speeds, t_step, time):
        # The true motor speeds can not fall below min and max speeds
        rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Compute individual rotor thrusts and net thrust and net moment
        rotor_thrusts   = self.k_thrust * rotor_speeds**2
        TM              = self.to_TM @ rotor_thrusts
        T               = TM[0]
        M               = TM[1:4]

        # Form autonomous ODE for constant inputs and integrate one time step
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)

        ##################### Development code - Test and verify before use! #####################
        def s_dot_nom_fn(t, s):
            return self._s_dot_nom_fn(t, s, T, M)
        ##################### Development code - Test and verify before use! #####################

        states_dot = self._s_dot_fn(0, state, T, M)

        s       = state
        sol     = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s       = sol['y'][:, -1]

        # Re-normalize unit quaternion.
        s[6:10] = s[6:10] / norm(s[6:10])

        # print(s.shape)
        # print(self.state_noise)
        #TODO Add state_noise to s
        noise_vector = np.zeros_like(s)

        # Assign the noise to the first 6 elements of the noise vector
        noise_vector[:6] = self.state_noise[:6]

        # Add the noise vector to the state vector
        # s = s + noise_vector
        # s = s + self.state_noise[:6]

        ##################### Development code - Test and verify before use! #####################
        sol = scipy.integrate.solve_ivp(s_dot_nom_fn, (0, t_step), state, first_step=t_step)
        state_nom = sol['y'][:, -1]

        # Re-normalize unit quaternion.
        state_nom[6:10] = state_nom[6:10] / norm(state_nom[6:10])

        state_err = s - state_nom
        ##################### Development code - Test and verify before use! #####################

        self.cnt += 1
        return s, self.rotor_spd, states_dot, state_nom, state_err

    def _s_dot_fn(self, t, state, u1, u2):
        # Position derivative (inertial velocities)
        x_dot = state[3:6]

        # Velocity derivative
        F = u1 * self.rotate_k(state[6:10])

        # Adding rotor and fuselage drag terms
        if include_aerodrag:
            r                   = Rotation.from_quat(state[6:10])
            rot_mat             = r.as_matrix()
            vel_b               = rot_mat.T @ x_dot
            factor              = 0.3
            a_drag_fuselage     = -0.08*factor * vel_b**2 * np.sign(vel_b) / self.mass
            rotor_drag          = np.array([0.3*factor, 0.3*factor, 0.0])
            a_drag_rotor        = -rotor_drag * vel_b / self.mass
            total_a_drag        = rot_mat @ (a_drag_rotor+a_drag_fuselage)
            v_dot               = F / self.mass + np.array([0, 0, -self.g]) + total_a_drag
        else:
            v_dot               = F / self.mass + np.array([0, 0, -self.g])  # upwards is positive

        # Orientation derivative
        q_dot = self.quat_dot(state[6:10], state[10:13])

        # Angular velocity derivative
        omega = state[10:13]
        omega_hat = self.hat_map(omega)
        disturbance_torques = np.array([0, 0, 0.0])
        w_dot = self.inv_inertia @ (u2 + - omega_hat @ (self.inertia @ omega) + disturbance_torques)

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot

        return s_dot

    ##################### Development code - Test and verify before use! #####################
    def _s_dot_nom_fn(self, t, state, u1, u2):
        # Position derivative. (inertial velocities)
        x_dot = state[3:6]

        # Velocity derivative.
        F = u1 * self.rotate_k(state[6:10])

        # Adding rotor and fuselage drag terms.
        v_dot = F / self.mass_nom + np.array([0, 0, -self.g])  # upwards is positive

        # Orientation derivative.
        q_dot = self.quat_dot(state[6:10], state[10:13])

        # Angular velocity derivative.
        omega = state[10:13]
        omega_hat = self.hat_map(omega)
        w_dot = self.inv_inertia_nom @ (u2 + - omega_hat @ (self.inertia_nom @ omega))

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot

        return s_dot
    ##################### Development code - Test and verify before use! #####################

    def rotate_k(cls, q):
        return np.array([2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)])

    def hat_map(cls, s):
        return np.array([[0, -s[2],  s[1]],
                         [s[2],     0, -s[0]],
                         [-s[1],  s[0],     0]])

    def quat_dot(self, quat, omega):
        """
        Parameters:
            quat, [i,j,k,w]
            omega, angular velocity of body in body axes
        Returns
            duat_dot, [i,j,k,w]
        """
        # Adapted from "Quaternions And Dynamics" by Basile Graf.
        (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
        G = np.array([[q3, q2, -q1, -q0],
                      [-q2, q3, q0, -q1],
                      [q1, -q0, q3, -q2]])
        quat_dot = 0.5 * G.T @ omega
        # Augment to maintain unit quaternion.
        quat_err = np.sum(quat ** 2) - 1
        quat_err_grad = 2 * quat
        quat_dot = quat_dot - quat_err * quat_err_grad
        return quat_dot