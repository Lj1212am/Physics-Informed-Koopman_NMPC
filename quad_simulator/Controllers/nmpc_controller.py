from casadi import *
import numpy as np
from quad_simulator.nmpc_utils import rk4, rotate_k, quat_dot, hat_map, quaternion_to_rotation_matrix_casadi


class QuadrotorNMPC:
    def __init__(self):
        self.T = 1
        self.N = 5

        self.mass = 0.03
        self.Ixx = 1.43e-5
        self.Iyy = 1.43e-5
        self.Izz = 2.89e-5

        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.inv_inertia = inv(self.inertia)
        self.g = 9.81

        self.DT = self.T / self.N
        self.state_dim = 13  # [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        self.control_dim = 4  # [Thrust, wx, wy, wz]

        self.arm_length = 0.046
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.k_thrust = 2.3e-08
        self.k_drag = 7.8e-11

        k = self.k_drag / self.k_thrust
        self.ctrl_forces_map = np.array([[1, 1, 1, 1],
                                         [0, self.arm_length, 0, -self.arm_length],
                                         [-self.arm_length, 0, self.arm_length, 0],
                                         [k, -k, k, -k]])
        self.forces_ctrl_map = np.linalg.inv(self.ctrl_forces_map)
        self.trim_motor_spd = 1790.0
        trim_force = self.k_thrust * np.square(self.trim_motor_spd)
        self.forces_old = np.array([trim_force, trim_force, trim_force, trim_force])
        self.u_old = np.array([self.mass * self.g, 0, 0, 0])

    def system_dynamics(self, use_nom=False):
        x = MX.sym('x', self.state_dim)
        u = MX.sym('u', self.control_dim)

        mass = self.mass_nom if use_nom else self.mass
        Ixx = self.Ixx_nom if use_nom else self.Ixx
        Iyy = self.Iyy_nom if use_nom else self.Iyy
        Izz = self.Izz_nom if use_nom else self.Izz

        ## Set up ODE for quadrotor nominal dynamics
        rotateK = rotate_k()
        hatMap = hat_map()
        quatDot = quat_dot()

        q = x[6:10]
        omega = x[10:13]
        e3 = DM([0, 0, 1])

        r_mat_func = quaternion_to_rotation_matrix_casadi()
        R_mat = r_mat_func(q)


        # Position derivative (inertial velocities)
        x_dot = x[3:6]

        # Velocity derivative
        # F = u[0] * rotateK(q)  # Rotated thrust vector
        # v_dot = F / mass + vertcat(0, 0, -self.g)  # Acceleration
        F = u[0]
        v_dot = R_mat @ (1 / mass * F * e3) - self.g * e3

        # Orientation derivative

        # Correct construction of each row of G matrix for quaternion dynamics using horzcat
        # G_row1 = horzcat(-q[1], -q[2], -q[3])
        # G_row2 = horzcat(q[0], -q[3], q[2])
        # G_row3 = horzcat(q[3], q[0], -q[1])
        # G_row4 = horzcat(-q[2], q[1], q[0])
        # #
        # # # Stack the rows vertically to create the full G matrix
        # G = vertcat(G_row1, G_row2, G_row3, G_row4)
        # dq = 0.5 * mtimes(G,omega)

        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        G = vertcat(
            horzcat(q3, q2, -q1, -q0),
            horzcat(-q2, q3, q0, -q1),
            horzcat(q1, -q0, q3, -q2)
        )

        # Calculate the quaternion derivative
        dq = 0.5 * mtimes(G.T, omega)
        quat_err = sum1(q ** 2) - 1
        quat_err_grad = 2 * q
        q_dot = dq - quat_err * quat_err_grad


        # q_dot = quatDot(q, omega)

        # Angular velocity derivative
        n = 3  # Since we are dealing with a 3x3 inertia matrix
        M = u[1:4]

        # Initialize a 3x3 matrix with zeros
        J = MX.zeros(n, n)

        # Manually set the diagonal elements
        J[0, 0] = Ixx
        J[1, 1] = Iyy
        J[2, 2] = Izz

        I = MX.eye(n)  # Create an identity matrix of size 3x3

        # Compute the inverse of J by solving J * J_inv = I
        J_inv = solve(J, I)
        omega_hat = hatMap(omega)
        disturbance_torques = MX([0.0, 0.0, 0.0])
        w_dot = mtimes(J_inv, (u[1:4] + - mtimes(omega_hat, mtimes(J, omega)) + disturbance_torques))
        # w_dot = mtimes(J_inv, (u[1:4] - cross(omega, mtimes(J, omega)) + disturbance_torques))
        f = vertcat(x_dot, v_dot, q_dot, w_dot)

        # RK4 integration
        ode = Function('ode', [x, u], [f])

        return ode


    # def system_dynamics(self, use_nom=False):
    #     x = MX.sym('x', self.state_dim)
    #     u = MX.sym('u', self.control_dim)
    #
    #     mass = self.mass_nom if use_nom else self.mass
    #     Ixx = self.Ixx_nom if use_nom else self.Ixx
    #     Iyy = self.Iyy_nom if use_nom else self.Iyy
    #     Izz = self.Izz_nom if use_nom else self.Izz
    #
    #     q = x[6:10]
    #     omega = x[10:13]
    #
    #     ## Set up ODE for quadrotor nominal dynamics
    #     rotateK = rotate_k()
    #     hatMap = hat_map()
    #     quatDot = quat_dot()
    #     # rot2quat = quaternion_to_rotation_matrix()
    #
    #     # Position derivative (inertial velocities)
    #     e3 = DM([0, 0, 1])
    #
    #     # States
    #     v = x[3:6]
    #     # R_mat = quaternion_to_rotation_matrix(q)
    #     # r_mat_func = quaternion_to_rotation_matrix_casadi()
    #     # R_mat = r_mat_func(q)
    #
    #     # Correct construction of each row of G matrix for quaternion dynamics using horzcat
    #     G_row1 = horzcat(-q[1], -q[2], -q[3])
    #     G_row2 = horzcat(q[0], -q[3], q[2])
    #     G_row3 = horzcat(q[3], q[0], -q[1])
    #     G_row4 = horzcat(-q[2], q[1], q[0])
    #
    #     # Stack the rows vertically to create the full G matrix
    #     G = vertcat(G_row1, G_row2, G_row3, G_row4)
    #
    #     # Calculate quaternion derivative
    #     dq = 0.5 * G @ omega
    #
    #     # Quaternion error and correction to ensure normalization
    #     quat_err = sum1(q ** 2) - 1
    #     quat_err_grad = 2 * q
    #     dq = dq - quat_err * quat_err_grad
    #
    #     # Control inputs
    #     # F = u[0]
    #     M = u[1:4]
    #
    #     # Dynamics
    #     # a = R_mat @ (1 / mass * F * e3) - self.g * e3
    #     # dR = R_mat @ hatMap(omega)
    #     F = u[0] * rotateK(q)  # Rotated thrust vector
    #     a = F / mass + vertcat(0, 0, -self.g)  # Acceleration
    #
    #     # Manually set the diagonal elements
    #     J = MX.zeros(3, 3)
    #     J[0, 0] = Ixx
    #     J[1, 1] = Iyy
    #     J[2, 2] = Izz
    #
    #     I = MX.eye(3)  # Create an identity matrix of size 3x3
    #     #
    #     #     # Compute the inverse of J by solving J * J_inv = I
    #     J_inv = solve(J, I)
    #
    #
    #
    #     # Note: The quaternion error correction is omitted as we're not directly working with quaternions here.
    #
    #     omega_hat = hatMap(omega)
    #     omega_dot = solve(J, M - omega_hat @ (J_inv @ omega))
    #
    #     # # Constructing the derivative of the state vector
    #     x_dot = vertcat(v, a, dq, omega_dot)
    #     # w_dot = mtimes(J_inv, (u[1:4] - cross(omega, mtimes(J, omega)) + disturbance_torques))
    #     # f = vertcat(x_dot, v_dot, q_dot, w_dot)
    #
    #     # RK4 integration
    #     ode = Function('ode', [x, u], [x_dot])
    #
    #     return ode

    def update(self, time, state, ref):
        # Update parameter values instead of resetting Opti

        opti = Opti()

        X = opti.variable(self.state_dim, self.N + 1)
        U = opti.variable(self.control_dim, self.N)

        ode = self.system_dynamics()

        rk4_step = rk4(ode, self.DT, self.state_dim, self.control_dim)

        for k in range(self.N):
            x_next = rk4_step(X[:, k], U[:, k])
            # print(x_next)
            # Impose the constraint that the next state in the sequence matches this updated state
            opti.subject_to(X[:, k + 1] == x_next)

        # # Convert ref to the full target state if ref has less dimensions
        # target_state = ref + [0] * (self.state_dim - len(ref))  # Adjust if ref structure differs
        pos_des = ref[0:3]
        vel_des = ref[3:6]

        # Extract desired yaw from ref, assuming ref[15] is the yaw value
        yaw_des = ref[15]

        # Convert yaw to quaternion
        qw = np.cos(yaw_des / 2.0)
        # print('qw', qw)
        qx = 0  # Rotation around z-axis, so qx = 0
        qy = 0  # Rotation around z-axis, so qy = 0
        qz = np.sin(yaw_des / 2.0)
        # print('qz', qz)
        # qx, qy, qz, qw = safe_normalize([qx, qy, qz, qw])

        # Construct the target state for NMPC
        target_state = np.concatenate((pos_des, vel_des, [qx, qy, qz, qw], [0, 0, 0]))  # Append zero angular velocities
        # print('quaternion state passed in', state[6:10])
        # q_norm = safe_normalize(state[6:10])
        # print('q_norm:', q_norm)
        # state[6:10] = q_norm
        # print('is quat normalized?:', sum(abs(state[6:10])) == 1)
        # print('resulting yaw:', quaternion_to_yaw(state[6:10]))
        # print('target state quaternions', target_state[6:10])

        # Re-apply constraints and objective with new initial and target states
        opti.subject_to(X[:, 0] == state)
        opti.set_initial(X[:, 0], state)
        opti.set_initial(U[:, 0], self.u_old)
        # opti.subject_to(X[:, -1]      == target_state)
        opti.subject_to(opti.bounded(-1, U, 1))

        opti.minimize(sumsqr(X - target_state) + sumsqr(U))

        opts = {'ipopt': {
            'max_iter': 1000,
            'print_level': 0,
            #'hessian_approximation': 'limited-memory',
            # 'sb': 'yes'},
            },
            'print_time': 0}
        # }
        # opts["monitor"] = ['nlp_g']
        # IpoptSolver(sb="yes")
        # opts['sb'] = "yes"

        opti.solver('ipopt', opts)

        try:
            solution = opti.solve()  # Solve the NMPC problem
            control = solution.value(U)


            control = control[:, 0]

            print(control)
            self.u_old = control

            forces = self.forces_ctrl_map @ control
            # forces = forces  # * 2.0
            # print('forces', forces)
            # Protect against invalid force and motor speed commands (set them to previous motor speeds)
            forces[forces < 0] = np.square(self.forces_old[forces < 0]) * self.k_thrust
            cmd_motor_speeds = np.sqrt(forces / self.k_thrust)
            self.forces_old = forces

            # print('cmd_motor_spds', cmd_motor_speeds)

            # print('cmd_motor_spds', cmd_motor_speeds)

            # Software limits for motor speeds
            cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

            return cmd_motor_speeds
        except RuntimeError as e:
            print("Solver failed:", str(e))
            # Inspect the values of different variables
            print("Last values of state variables:", opti.debug.value(X))
            print("Last values of control inputs:", opti.debug.value(U))
            opti.debug.show_infeasibilities()
