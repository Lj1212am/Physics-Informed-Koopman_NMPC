import numpy as np
from casadi import *
from scipy.spatial.transform import Rotation as R

def quat_dot_np(quat, omega):
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[q3, q2, -q1, -q0],
                  [-q2, q3, q0, -q1],
                  [q1, -q0, q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    quat_err = np.sum(quat ** 2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

def quat_dot_casadi(quat, omega):
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    G = vertcat(
        horzcat(q3, q2, -q1, -q0),
        horzcat(-q2, q3, q0, -q1),
        horzcat(q1, -q0, q3, -q2)
    )
    quat_dot = 0.5 * mtimes(G.T, omega)
    quat_err = sum1(quat ** 2) - 1
    quat_err_grad = 2 * quat
    quat_dot -= quat_err * quat_err_grad
    return quat_dot


    # NumPy Implementation for computing J_inv

def compute_J_inv_numpy(Ixx, Iyy, Izz):
    inertia = np.diag(np.array([Ixx, Iyy, Izz]))
    inv_inertia = np.linalg.inv(inertia)
    return inv_inertia

# CasADi Implementation for computing J_inv
def compute_J_inv_casadi(Ixx, Iyy, Izz):
    n = 3
    J = MX.zeros(n, n)
    J[0, 0] = Ixx
    J[1, 1] = Iyy
    J[2, 2] = Izz
    I = MX.eye(n)
    J_inv = solve(J, I)
    # Define inputs and outputs for the CasADi function
    Ixx_sym, Iyy_sym, Izz_sym = MX.sym('Ixx'), MX.sym('Iyy'), MX.sym('Izz')

    # Instead of an empty input list, use the symbolic variables as inputs
    J_inv_func = Function('J_inv_func', [Ixx_sym, Iyy_sym, Izz_sym], [J_inv])

    # Now, call the function with actual values to evaluate J_inv
    J_inv_numeric = J_inv_func(Ixx, Iyy, Izz)  # Pass the actual inertia values

    # Access the first (and only) output and convert it to a NumPy array
    return J_inv_numeric.full()

def hat_map_casadi_sym():
    s = MX.sym('s', 3)  # Define a symbolic vector of size 3
    S = vertcat(
        horzcat(0, -s[2], s[1]),
        horzcat(s[2], 0, -s[0]),
        horzcat(-s[1], s[0], 0)
    )
    # Create a CasADi function from symbolic inputs to the skew-symmetric matrix
    return Function('hat_map', [s], [S])

# Create the CasADi function instance
hat_map_casadi_func = hat_map_casadi_sym()

# NumPy implementation remains the same
def hat_map_numpy(s):
    return np.array([[0, -s[2],  s[1]],
                     [s[2],     0, -s[0]],
                     [-s[1],  s[0],     0]])

def rotate_k_casadi_sym():
    q = MX.sym('q', 4)  # Define a symbolic quaternion
    k_vector = vertcat(
        2.0 * (q[0] * q[2] + q[1] * q[3]),
        2.0 * (q[1] * q[2] - q[0] * q[3]),
        1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
    )
    return Function('rotate_k', [q], [k_vector])

def rotate_k_numpy(q):
    return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                     2 * (q[1] * q[2] - q[0] * q[3]),
                     1 - 2 * (q[0]**2  + q[1]**2)])

# CasADi function definition
quat_sym = MX.sym('quat', 4)
omega_sym = MX.sym('omega', 3)
quat_dot_casadi_sym = quat_dot_casadi(quat_sym, omega_sym)
quat_dot_fun = Function('quat_dot_fun', [quat_sym, omega_sym], [quat_dot_casadi_sym])

# Unit tests
test_cases = [
    (np.array([1, 0, 0, 0]), np.array([0, 0, 0])),  # No rotation
    (np.array([0, 1, 0, 0]), np.array([1, 0, 0])),  # Rotation around x-axis
    (np.array([0, 0, 1, 0]), np.array([0, 1, 0])),  # Rotation around y-axis
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1])),  # Rotation around z-axis
]

for quat, omega in test_cases:
    # Evaluate the CasADi function
    quat_dot_casadi_result = quat_dot_fun(quat, omega).full().flatten()

    # Directly use the provided NumPy implementation for comparison
    quat_dot_np_result = quat_dot_np(quat, omega)

    # Output results for comparison
    print(f"Testing quat: {quat}, omega: {omega}")
    print(f"NumPy result: {quat_dot_np_result}")
    print(f"CasADi result: {quat_dot_casadi_result}")

    # Assert that the results are close
    assert np.allclose(quat_dot_np_result, quat_dot_casadi_result, atol=1e-6), "Mismatch between NumPy and CasADi results"




# Test cases for J_inv computation
test_cases = [
    (1.0, 1.0, 1.0),  # Equal inertia
    (0.01, 0.02, 0.03),  # Typical condition
    (10, 1, 0.1),  # Extreme ratios
    (0.001, 0.001, 0.001),  # Minimal inertia
]

for Ixx, Iyy, Izz in test_cases:
    # Compute J_inv using both CasADi and NumPy
    J_inv_casadi = compute_J_inv_casadi(Ixx, Iyy, Izz)  # Convert to NumPy array for comparison
    J_inv_numpy = compute_J_inv_numpy(Ixx, Iyy, Izz)
    print('casadi', J_inv_casadi)
    print('np', J_inv_numpy)

    # Verify that CasADi and NumPy results match
    assert np.allclose(J_inv_casadi, J_inv_numpy), f"Mismatch in J_inv for Ixx={Ixx}, Iyy={Iyy}, Izz={Izz}"

    # Additional validation: J * J_inv should equal the identity matrix
    J = np.diag([Ixx, Iyy, Izz])
    identity_casadi = np.dot(J, J_inv_casadi)
    identity_numpy = np.dot(J, J_inv_numpy)

    assert np.allclose(identity_casadi, np.eye(3)), "CasADi J_inv does not correctly invert J"
    assert np.allclose(identity_numpy, np.eye(3)), "NumPy J_inv does not correctly invert J"

# Test vectors
test_vectors = [
    np.array([1, 2, 3]),
    np.array([-1, -2, -3]),
    np.array([0, 0, 0]),
    np.array([10, -5, 2.5])
]

for s in test_vectors:
    # Compute skew-symmetric matrix using CasADi function with numeric inputs
    S_casadi = hat_map_casadi_func(s).full()  # Evaluate with numeric input

    # Compute skew-symmetric matrix using NumPy
    S_numpy = hat_map_numpy(s)

    # Verify skew-symmetry for NumPy result
    assert np.allclose(-S_numpy, S_numpy.T), "NumPy: Result is not skew-symmetric"

    # Compare CasADi and NumPy results
    assert np.allclose(S_casadi, S_numpy), f"Mismatch between CasADi and NumPy results for vector {s}"

rotate_k_casadi_func = rotate_k_casadi_sym()

# Test quaternions
test_quaternions = [
    np.array([0, 0, 0, 1]),  # No rotation
    np.array([0.7071, 0, 0, 0.7071]),  # 90 degrees rotation around x-axis
    np.array([0, 0.7071, 0, 0.7071]),  # 90 degrees rotation around y-axis
    np.array([0, 0, 0.7071, 0.7071]),  # 90 degrees rotation around z-axis
    np.array([0.5, 0.5, 0.5, 0.5])  # Rotation around x=y=z axis
]

for q in test_quaternions:
    # Compute rotation vector using CasADi function with numeric inputs
    k_vector_casadi = rotate_k_casadi_func(q).full().flatten()  # Evaluate with numeric input

    # Compute rotation vector using NumPy
    k_vector_numpy = rotate_k_numpy(q)

    # Compare CasADi and NumPy results
    assert np.allclose(k_vector_casadi, k_vector_numpy), f"Mismatch between CasADi and NumPy results for quaternion {q}"



def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a rotation matrix.
    quat: Quaternion vector [x, y, z, w] represented as a CasADi MX type,
          where w is the scalar part.
    """
    # Reordering quat to correctly place the scalar part as the fourth element

    # x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    # # Calculate the norm of the quaternion
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # x, y, z, w = normalized_q[0], normalized_q[1], normalized_q[2], normalized_q[3]

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

def quaternion_to_rotation_matrix_numpy(quat):
    # Convert quaternion to rotation matrix using scipy
    rotation = R.from_quat(quat)
    return rotation.as_matrix()
    # # Unpack the quaternion
    # x, y, z, w = quat
    #
    # # Compute the rotation matrix elements
    # r00 = 1 - 2 * (y ** 2 + z ** 2)
    # r01 = 2 * (x * y - z * w)
    # r02 = 2 * (x * z + y * w)
    # r10 = 2 * (x * y + z * w)
    # r11 = 1 - 2 * (x ** 2 + z ** 2)
    # r12 = 2 * (y * z - x * w)
    # r20 = 2 * (x * z - y * w)
    # r21 = 2 * (y * z + x * w)
    # r22 = 1 - 2 * (x ** 2 + y ** 2)
    #
    # rot_mat = np.zeros((3, 3))
    # rot_mat[0, 0] = r00
    # rot_mat[0, 1] = r01
    # rot_mat[0, 2] = r02
    # rot_mat[1, 0] = r10
    # rot_mat[1, 1] = r11
    # rot_mat[1, 2] = r12
    # rot_mat[2, 0] = r20
    # rot_mat[2, 1] = r21
    # rot_mat[2, 2] = r22
    # # # Create and return the rotation matrix
    # # rot_matrix = np.array([[r00, r01, r02],
    # #                        [r10, r11, r12],
    # #                        [r20, r21, r22]])
    #
    # return rot_mat


# Define a CasADi Function for the quaternion to rotation matrix conversion
q_sym = MX.sym('q', 4)
R_sym_casadi = quaternion_to_rotation_matrix(q_sym)
quat_to_rot_casadi_func = Function('quat_to_rot_casadi', [q_sym], [R_sym_casadi])

# Test cases
test_quaternions = [
    np.array([1, 0, 0, 0]),  # Identity rotation
    np.array([0, 1, 0, 0]),  # 180 degrees about x-axis
    np.array([0, 0, 1, 0]),  # 180 degrees about y-axis
    np.array([0, 0, 0, 1]),  # 180 degrees about z-axis
    np.array([0.7071, 0.7071, 0, 0]),  # 90 degrees about x-axis
    # Add more test cases as needed
]

# Tolerance for comparison
tolerance = 1e-5

# Testing loop
for quat in test_quaternions:
    # Evaluate the CasADi function
    R_casadi_evaluated = quat_to_rot_casadi_func(DM(quat)).full()

    # Evaluate the NumPy function
    R_numpy_evaluated = quaternion_to_rotation_matrix_numpy(quat)

    # Compare results
    assert np.allclose(R_casadi_evaluated, R_numpy_evaluated, atol=tolerance), f"Mismatch between CasADi and NumPy results for quaternion {quat}. CasADi rotation matrix:\n{R_casadi_evaluated}\nNumPy rotation matrix:\n{R_numpy_evaluated}"
print('all quats rotated correctly')

# def psi_numpy(X, n_basis=3):
#     quat = X[6:10]
#     print('X', X)
#     R = quaternion_to_rotation_matrix_numpy(quat)
#     print(R)
#     positional_states = X[:3]
#     print('ps', positional_states)
#     velocity_states = X[3:6]
#     print('vs', velocity_states)
#     wb = X[10:13]
#     wb_hat = hat_map_numpy(wb)
#
#     basis = np.zeros((9 * n_basis, 1))
#     Z = R
#     for i in range(n_basis):
#         Z = np.dot(Z, wb_hat)  # Matrix multiplication in NumPy
#         basis[9 * i:9 * (i + 1), :] = Z.reshape((9, 1), order="F")
#
#     R_reshaped = R.reshape((9, 1),order="F")
#     basis_result = np.vstack([R_reshaped, wb.reshape((3, 1), ), quat.reshape((4, 1), order="F"), basis])
#     translational_result = np.vstack([positional_states.reshape((3, 1), order="F"), velocity_states.reshape((3, 1), order="F")])
#
#     # lifted_states = np.vstack([positional_states.reshape((3, 1)), velocity_states.reshape((3, 1)), basis_result])
#     lifted_states = np.vstack([translational_result, basis_result])
#
#
#     return lifted_states

def psi_numpy(X, n_basis=3):
    # Extract states
    Q = X[6:10]
    # print('X', X)
    R = quaternion_to_rotation_matrix_numpy(Q)
    wb = X[10:13].reshape(3, 1)
    wb = wb.squeeze()
    Q = Q.reshape(4, 1)
    wb_hat = hat_map_numpy(wb)

    positional_states = X[:3]
    velocity_states = X[3:6]

    # Build observables
    basis = np.zeros((9 * n_basis, 1))
    Z = R
    for i in range(n_basis):
        Z = np.dot(Z, wb_hat)
        basis[i * 9:(i + 1) * 9, :] = Z.flatten(order='F').reshape(9, 1)

    print(wb.reshape(-1, 1).shape)
    # Concatenate results
    basis_result = np.concatenate([R.flatten(order='F').reshape(9, 1), wb.reshape(-1, 1), Q, basis])
    translational_result = np.vstack(
        [positional_states.reshape((3, 1), order="F"), velocity_states.reshape((3, 1), order="F")])

    lifted_states = np.vstack([translational_result, basis_result])

    return lifted_states


# Execute the test
# test_psi_numpy()


def psi_casadi(X, n_basis=3):
    # Adjusted to consider quaternion scalar part as the 4th element.
    # Extract quaternion and angular velocity from the state X
    # X is assumed to be ordered as [translational; quaternion; angular velocity]
    quat = X[6:10]  # quaternion in the form [qx, qy, qz, qw]
    R = quaternion_to_rotation_matrix(quat)
    quaternion = reshape(X[6:10], (4, 1))
    positional_states = X[:3]  # Extract positional xyz states
    velocity_states = X[3:6]  # Extract velocity xyz states

    wb = reshape(X[10:13], (3, 1))  # angular velocity
    # wb = X[10:13]

    hm = hat_map_casadi_sym()
    wb_hat = hm(wb)

    basis = MX.zeros(9 * n_basis, 1)
    Z = R
    for i in range(n_basis):
        Z = mtimes(Z, wb_hat)  # Update Z based on angular velocity
        basis[9 * i:9 * (i + 1), :] = reshape(Z, (9, 1))  # Store the flattened Z in the basis vector

    # Flatten and reshape R to a 9x1 vector, assuming Fortran-style (column-major) order
    R_reshaped = reshape(R, (9, 1))

    # Concatenate all parts to form the basis_result vector
    basis_result = vertcat(R_reshaped, wb, quaternion, basis)

    # Vertically concatenate the positional states, velocity states, and the basis into a single column vector
    lifted_states = vertcat(positional_states, velocity_states, basis_result)

    # Return the basis containing the quaternion, angular velocity, and higher-order terms
    return lifted_states


def test_psi_functions():
    X = MX.sym('X', 13)  # Assuming X has 13 elements
    lifted_states = psi_casadi(X)  # Call your existing psi_casadi logic here
    psi_casadi_func = Function('psi_casadi', [X], [lifted_states], ['X'], ['lifted_states'])

    # Define test inputs (adjust based on your scenario)
    test_inputs = [
        np.array([1, 2, 3, 0.1, 0.2, 0.3, 0, 0, 0, 1, 0.1, 0.2, 0.3]),
        np.array([2, 3, 4, 0.2, 0.3, 0.4, 0.7071, 0, 0, 0.7071, 0.2, 0.3, 0.4]),
    ]

    for test_input in test_inputs:
        # Calculate lifted state using NumPy-based psi function
        lifted_state_numpy = psi_numpy(test_input)

        # Convert test_input to CasADi DM type for compatibility with psi_casadi
        # Convert test_input to CasADi type (DM) for compatibility with psi_casadi function
        test_input_casadi = DM(test_input)

        # Calculate lifted state using CasADi-based psi function
        # Ensure that psi_casadi is correctly defined as a CasADi Function


        lifted_state_casadi = psi_casadi_func(X=test_input_casadi)['lifted_states'].full().flatten()

        # Compare CasADi and NumPy results
        # Check if the outputs match
        if not np.allclose(lifted_state_numpy.flatten(), lifted_state_casadi, atol=1e-5):
            print(f"Mismatch between CasADi and NumPy results for input {test_input}")
            print("NumPy result:", lifted_state_numpy.flatten()[22:])
            print("CasADi result:", lifted_state_casadi[22:])
            assert False, "Mismatch found!"


    print("All psi function tests passed.")


# Make sure to define psi_casadi as a CasADi function here
# psi_casadi = Function('psi_casadi', [X], [psi_logic_here], ['X'], ['lifted_state'])

# Execute the test
test_psi_functions()


# Define the sizes of the states
len_z = 22  # Size of the lifted state
nx = 13  # Size of the original state

# Initialize the C matrix
# Given the structure, we understand the mapping directly from your description
C = np.zeros((nx, len_z))

# Assuming the lifted state is structured as follows:
# First 3: Position, Next 3: Velocity, 9: Placeholder for rotation matrix (not extracted), Next 3: Angular velocity, Last 4: Quaternion
# Position and Velocity are directly mapped
# Initialize the C matrix with zeros


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

# Create a mock lifted state vector with known values for testing
# Using sequential values for easy verification
z = np.arange(1, len_z + 1)

# Apply the C matrix to map from the lifted state to the original state
x_mapped = C @ z

print('xmap', x_mapped)



print("All tests passed successfully!")