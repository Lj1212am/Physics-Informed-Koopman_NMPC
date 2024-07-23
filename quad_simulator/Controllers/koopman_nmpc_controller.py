from casadi import *
import numpy as np
from quad_simulator.nmpc_utils import rotation_matrix_to_quaternion, axis_angle_to_quaternion, vee_map, rk4, rotate_k, \
    quat_dot, hat_map, quaternion_to_rotation_matrix
from scipy.linalg import logm


class QuadrotorKoopNMPC:
    def __init__(self):

        self.lifted_state = None
        self.training_method = None
        self.B_err = None
        self.A_err = None
        self.C = None
        self.B_nom = None
        self.A_nom = None
        self.B = None
        self.A = None
        self.T = 1
        self.N = 5

        self.mass = 0.03
        self.Ixx = 1.43e-5
        self.Iyy = 1.43e-5
        self.Izz = 2.89e-5

        self.mass_nom = 0.03 * 2.0
        self.Ixx_nom = 1.43e-5 * 1.5
        self.Iyy_nom = 1.43e-5 * 1.5
        self.Izz_nom = 2.89e-5 * 1.2

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

    def set_koopman(self, training_method, A=None, B=None, C=None, A_nom=None, B_nom=None, A_err=None, B_err=None):
        self.training_method = training_method

        self.A = A
        # inf_mask_A = np.isinf(A)
        # # Check for Inf values
        #
        # if np.any(inf_mask_A):
        #     print("A Matrix contains Inf values.")
        # else:
        #     print("A Matrix does not contain Inf values.")

        self.B = B
        # inf_mask_B = np.isinf(B)

        # if np.any(inf_mask_B):
        #     print("B Matrix contains Inf values.")
        # else:
        #     print("B Matrix does not contain Inf values.")
        # print(C.shape)

        self.A_nom = A_nom
        self.B_nom = B_nom
        self.A_err = A_err
        self.B_err = B_err

        n_basis = 3  # Number of basis functions
        nx = 13  # Number of original state components
        len_z = 3 + 3 + 9 + 3 + 4 + 9 * n_basis  # Update based on the actual structure

        self.lifted_state = len_z

        # Initialize the C matrix with zeros
        C = np.zeros((nx, len_z))

        # Extract positional and velocity states directly
        C[0:3, 0:3] = np.eye(3)  # Position
        C[3:6, 3:6] = np.eye(3)  # Velocity

        # Extract angular velocity (wb) and quaternion from their known positions
        # The angular velocity comes right after the rotation matrix
        ang_vel_start_index = 6 + 9  # Starting right after the flattened rotation matrix
        C[10:13, ang_vel_start_index:ang_vel_start_index + 3] = np.eye(3)  # Angular velocity

        # The quaternion comes right after angular velocity
        quat_start_index = ang_vel_start_index + 3
        C[6:10, quat_start_index:quat_start_index + 4] = np.eye(4)  # Quaternion

        self.C = C

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

        # Position derivative (inertial velocities)
        x_dot = x[3:6]

        # Velocity derivative
        F = u[0] * rotateK(q)  # Rotated thrust vector
        v_dot = F / mass + vertcat(0, 0, -self.g)  # Acceleration

        # Orientation derivative
        q_dot = quatDot(q, omega)

        # Angular velocity derivative
        n = 3  # Since we are dealing with a 3x3 inertia matrix

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
        # w_dot = mtimes(J_inv, (u[1:4] + - mtimes(omega_hat, mtimes(J, omega)) + disturbance_torques))
        w_dot = mtimes(J_inv, (u[1:4] - cross(omega, mtimes(J, omega)) + disturbance_torques))
        f = vertcat(x_dot, v_dot, q_dot, w_dot)

        # RK4 integration
        ode = Function('ode', [x, u], [f])

        return ode

    # def koopman_to_quad_state_space(self, X_22):
    #
    #     # Extract components from the correction vector in the lifted space
    #     pos_corr = X_22[0:3]
    #     vel_corr = X_22[3:6]
    #     # R_flat_corr = X_22[6:15]  # This segment represents the flattened rotation matrix
    #
    #     # Convert the flattened rotation matrix back to a quaternion
    #     # R = (reshape(R_flat_corr, (3, 3)))
    #     # axis_angle = vee_map(logm(R))
    #     # quat_corr = axis_angle_to_quaternion(axis_angle)
    #
    #     # quat_corr = rotation_matrix_to_quaternion(R)
    #
    #     quat_corr = X_22[18:22]
    #     ang_vel_corr = X_22[15:18]
    #
    #     # Concatenate corrected components to form the corrected 13-dimensional state
    #     X_13 = vertcat(pos_corr, vel_corr, quat_corr, ang_vel_corr)
    #
    #     return X_13

    def psi(self, X, n_basis=3):
        # Adjusted to consider quaternion scalar part as the 4th element.
        # Extract quaternion and angular velocity from the state X
        # X is assumed to be ordered as [translational; quaternion; angular velocity]
        quat = X[6:10]  # quaternion in the form [qx, qy, qz, qw]
        R = quaternion_to_rotation_matrix(quat)
        quaternion = reshape(X[6:10], (4, 1))
        positional_states = X[:3]  # Extract positional xyz states
        velocity_states = X[3:6]  # Extract velocity xyz states

        # wb = reshape(X[10:13], (3, 1))  # angular velocity
        wb = X[10:13]

        hm = hat_map()
        wb_hat = hm(wb)

        basis = MX.zeros(9 * n_basis, 1)
        Z = R
        for i in range(n_basis):
            Z = mtimes(Z, wb_hat)  # Update Z based on angular velocity
            basis[9 * i:9 * (i + 1), :] = reshape(Z, (9, 1))  # Store the flattened Z in the basis vector

        # Flatten and reshape R to a 9x1 vector, assuming Fortran-style (column-major) order
        R_reshaped = reshape(R, 9, 1)

        # Concatenate all parts to form the basis_result vector
        basis_result = vertcat(R_reshaped, wb, quaternion, basis)

        # Vertically concatenate the positional states, velocity states, and the basis into a single column vector
        lifted_states = vertcat(positional_states, velocity_states, basis_result)

        # Return the basis containing the quaternion, angular velocity, and higher-order terms
        return lifted_states

    def update(self, time, state, ref):
        opti = Opti()
        X = opti.variable(self.state_dim, self.N + 1)
        U = opti.variable(self.control_dim, self.N)

        # Define additional CasADi variable for the nominal state
        # X_nom = opti.variable(self.state_dim, self.N + 1)

        # Z = opti.variable(self.lifted_state, self.N + 1)
        # initial_state = DM(state)  # Convert state to DM if it's a numpy array
        # opti.set_initial(X_nom, np.tile(initial_state, (1, self.N + 1)))  # Set initial guess for the nominal states

        ode = self.system_dynamics()
        ode_nom = self.system_dynamics(use_nom=True)

        rk4_step = rk4(ode, self.DT, self.state_dim, self.control_dim)
        rk4_nom = rk4(ode_nom, self.DT, self.state_dim, self.control_dim)
        constraint_expressions = []

        # for k in range(self.N):
        #     Z[:, k + 1] = mtimes(self.A, Z[:, k]) + mtimes(self.B, U[:, k])

        for k in range(self.N):

            # x_next = rk4_step(X[:, k], U[:, k])

            # x_nom_next = rk4_nom(X_nom[:, k], U[:, k])  # Nominal state is propagated with the same control inputs
            x_nom_next = rk4_nom(X[:, k], U[:, k])

            # Choose the dynamics to use based on the training method
            if self.training_method == 'pure_data':

                x_next = mtimes(self.C, mtimes(self.A, self.psi(X[:, k])) + mtimes(self.B, U[:, k]))
                # x_next = mtimes(self.C, Z[:, k])



            elif self.training_method == 'hybrid1':

                correction = mtimes(self.C,
                                    (mtimes(self.A, self.psi(X[:, k])) - mtimes(self.A_nom, self.psi(x_nom_next))) + \
                                    mtimes((self.B - self.B_nom), U[:, k]))

                x_next = x_nom_next + correction

            elif self.training_method == 'hybrid2':
                # Apply correction to account for error dynamics
                correction = mtimes(self.C, mtimes(self.A_err, self.psi(X[:, k])) + mtimes(self.B_err, U[:, k]))

                x_next = x_nom_next + correction

            # Impose the constraints for the true and nominal states

            opti.subject_to(X[:, k + 1] == x_next)
            constraint_expression = X[:, k + 1] - x_next  # This represents the expression you're constraining to be zero
            constraint_expressions.append(constraint_expression)

            # opti.subject_to(X_nom[:, k + 1] == x_nom_next)


        # # Convert ref to the full target state if ref has less dimensions
        pos_des = ref[0:3]
        vel_des = ref[3:6]

        # Extract desired yaw from ref, assuming ref[15] is the yaw value
        yaw_des = ref[15]

        # Convert yaw to quaternion
        qw = np.cos(yaw_des / 2.0)
        qx = 0  # Rotation around z-axis, so qx = 0
        qy = 0  # Rotation around z-axis, so qy = 0
        qz = np.sin(yaw_des / 2.0)

        # Construct the target state for NMPC
        target_state = np.concatenate((pos_des, vel_des, [qx, qy, qz, qw], [0, 0, 0]))

        # Re-apply constraints and objective with new initial and target states
        opti.subject_to(X[:, 0] == state)
        # opti.set_initial(X[:, 0], state)
        # opti.set_initial(U[:, 0], self.u_old)
        # opti.subject_to(X[:, -1]      == target_state)
        opti.subject_to(opti.bounded(-1, U, 1))

        opti.minimize(sumsqr(X - target_state) + sumsqr(U))

        opts = {'ipopt': {
            'max_iter': 1000,
            'print_level': 0,
            'tol': 1e-6,
        }
            # },
            , 'print_time': 0}
        # }
        # opts["monitor"] = ['nlp_g']
        # IpoptSolver(sb="yes")
        # opts['sb'] = "yes"

        opti.solver('ipopt', opts)

        try:
            solution = opti.solve()  # Solve the NMPC problem
            control = solution.value(U)

            control = control[:, 0]
            # control[0] = control[0] * 1e5
            print('control', control)

            self.u_old = control

            forces = self.forces_ctrl_map @ control
            # forces = forces  # * 2.0
            # print('forces', forces)
            # Protect against invalid force and motor speed commands (set them to previous motor speeds)
            forces[forces < 0] = np.square(self.forces_old[forces < 0]) * self.k_thrust
            cmd_motor_speeds = np.sqrt(forces / self.k_thrust)
            self.forces_old = forces

            print('cmd_motor_spds', cmd_motor_speeds)

            # Software limits for motor speeds
            cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

            return cmd_motor_speeds
        except RuntimeError as e:
            print("Solver failed:", str(e))
            # Inspect the values of different variables
            print("Last values of state variables:", opti.debug.value(X))
            # print("Last values of control inputs:", opti.debug.value(U))
            print("Debugging x_next:", opti.debug.value(x_next))
            # print("Variable X:", opti.debug.value(X))
            debug_index = 0  # Change this index to debug different iterations
            if debug_index < len(constraint_expressions):
                print(f"Debugging constraint expression at k={debug_index}:",
                      opti.debug.value(constraint_expressions[debug_index]))
            else:
                print(f"No constraint expression stored for k={debug_index}")
            opti.debug.show_infeasibilities()