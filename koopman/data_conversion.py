from imports import *

# Load the koopman data
data_file_path = '../quad_simulator/koopman_dat_err_wxyz.mat'
koopman_data = loadmat(data_file_path)

# Assuming 'data' is the main dataset and 'data_nom' is the nominal dataset
x = koopman_data['data']
x_nominal = koopman_data['data_nom']
x_error = koopman_data['data_err']


# Define the number of trajectories and the length of each trajectory
num_traj = 1  # Assuming 1 trajectories

# Calculate the length of each trajectory
traj_len = x.shape[0]
# per_sample = x.shape[1]

# Assuming each trajectory


# Initialize the matrices
X1 = np.zeros((22, num_traj * traj_len))
X2 = np.zeros((22, num_traj * traj_len))
X1nom = np.zeros((22, num_traj * traj_len))
X2nom = np.zeros((22, num_traj * traj_len))
X1err = np.zeros((22, num_traj * traj_len))
X2err = np.zeros((22, num_traj * traj_len))
U = np.zeros((4, num_traj * traj_len))  # Initialize U with zeros or default values

u = x[:, -4:].T


# Convert koopman data into train_dat format
for i in range(num_traj):
    start_idx = i * traj_len
    end_idx = (i + 1) * traj_len

    # Excluding the first row (time) and last four rows (control inputs) and reshaping the data

    X1[:, start_idx:end_idx] = x[: end_idx, 1:23].T
    X2[:, start_idx:end_idx - 1] = x[1:, 1:23].T
    X1nom[:, start_idx:end_idx] = x_nominal[:end_idx, 1: 23].T
    X2nom[:, start_idx:end_idx - 1] = x_nominal[1:, 1: 23].T
    X1err[:, start_idx:end_idx] = x_error[:end_idx, 1:23].T
    X2err[:, start_idx:end_idx - 1] = x_error[1:, 1:23].T

    # Extract control input U
    control_input = u[:, i]  # Extract the i-th row (single control input)
    U[:, start_idx:end_idx] = np.tile(control_input, (traj_len, 1)).T


data = {
    'X1': X1,
    'X2': X2,
    'X1nom': X1nom,
    'X2nom': X2nom,
    'X1err': X1err,
    'X2err': X2err,
    'U': U
}

# Splice and Save the converted data

# Define the split ratio
train_ratio = 0.8

# Calculate the split index
split_idx = int(train_ratio * traj_len * num_traj)

# Split the data for training and validation
train_data = {
    'X1': X1[:, :split_idx],
    'X2': X2[:, :split_idx],
    'X1nom': X1nom[:, :split_idx],
    'X2nom': X2nom[:, :split_idx],
    'X1err': X1err[:, :split_idx],
    'X2err': X2err[:, :split_idx],
    'U': U[:, :split_idx]
}

validation_data = {
    'X1': X1[:, split_idx:],
    'X2': X2[:, split_idx:],
    'X1nom': X1nom[:, split_idx:],
    'X2nom': X2nom[:, split_idx:],
    'X1err': X1err[:, split_idx:],
    'X2err': X2err[:, split_idx:],
    'U': U[:, split_idx:]
}

# Save the training data
scipy.io.savemat('train_data.mat', train_data)

# Save the validation data
scipy.io.savemat('validation_data.mat', validation_data)

# Save the training data
np.savez('train_data.npz', **train_data)

# Save the validation data
np.savez('validation_data.npz', **validation_data)
