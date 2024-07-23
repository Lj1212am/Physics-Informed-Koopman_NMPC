# Author: KongYao Chee (ckongyao@seas.upenn.edu)
# Code adapted from: https://github.com/sriram-2502/KoopmanMPC_Quadrotor
from imports import *
from EDMD import EDMD
from compute_rmse import compute_rmse
from construct_basis import construct_basis
from generate_dataset import generate_dataset
from quad_dynamics import quad_dynamics
from matrix_math import hat_map, vee_map

# Set random seed
np.random.seed(1)

# Options
training_method = 'pure_data'  # Options: 'pure_data', 'hybrid1', 'hybrid2'
# show_plot = False  # Plotting option (commented out as it's not available at the moment)

# Set initial parameters and conditions
dt = 1e-3
t_final = 0.1
t_span = np.arange(0, t_final + dt, dt)

# Initial state
p0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
R0 = np.eye(3)
w0 = np.array([0.1, 0, 0])
# Convert rotation matrix to quaternion using pyquaternion's Quaternion
q = Quaternion(matrix=R0.T).elements
# x0 = np.concatenate([p0, v0, R0.ravel(), w0, q])
x0 = np.concatenate([p0, v0, R0.ravel(), w0])
nx = len(x0)

## Generate new data
# Training dataset
n_control = 100
X1_train, X2_train, X1err_train, X2err_train, X1nom_train, X2nom_train, U_train = \
    generate_dataset(x0, n_control, t_span, 'train')
# Here we initialize 3D arrays to store rotation matrices and then fill them with reshaped data
R1_train = np.zeros((3, 3, X1_train.shape[1]))
R2_train = np.zeros((3, 3, X2_train.shape[1]))
R1nom_train = np.zeros((3, 3, X1nom_train.shape[1]))
R2nom_train = np.zeros((3, 3, X2nom_train.shape[1]))

for i in range(X1_train.shape[1]):
    R1_train[:, :, i] = X1_train[6:15, i].reshape(3, 3)
    R2_train[:, :, i] = X2_train[6:15, i].reshape(3, 3)
    R1nom_train[:, :, i] = X1nom_train[6:15, i].reshape(3, 3)
    R2nom_train[:, :, i] = X2nom_train[6:15, i].reshape(3, 3)
# Using pyquaternion to convert the rotation matrices to quaternions
# Q1_train = [Quaternion(matrix=R1_train[:, :, i]).elements for i in range(R1_train.shape[2])]
# Q2_train = [Quaternion(matrix=R2_train[:, :, i]).elements for i in range(R2_train.shape[2])]
# Q1nom_train = [Quaternion(matrix=R1nom_train[:, :, i]).elements for i in range(R1nom_train.shape[2])]
# Q2nom_train = [Quaternion(matrix=R2nom_train[:, :, i]).elements for i in range(R2nom_train.shape[2])]


# Validation dataset
n_control_val = 50
X1_val, X2_val, X1err_val, X2err_val, X1nom_val, X2nom_val, U_val = \
    generate_dataset(x0, n_control_val, t_span, 'val')
# Create dictionary for training data
train_dat = {
    "X1": X1_train,
    "X2": X2_train,
    "X1err": X1err_train,
    "X2err": X2err_train,
    "X1nom": X1nom_train,
    "X2nom": X2nom_train,
    "U": U_train
}

# Create dictionary for validation data
val_dat = {
    "X1": X1_val,
    "X2": X2_val,
    "X1err": X1err_val,
    "X2err": X2err_val,
    "X1nom": X1nom_val,
    "X2nom": X2nom_val,
    "U": U_val
}

# Save the dictionaries to .mat files
np.savez('train_dat.npz', **train_dat)
np.savez('val_dat.npz', **val_dat)


## Load training data from .mat file
# dat_train = scipy.io.loadmat('train_dat_wquat.mat')
# dat_train = scipy.io.loadmat('converted_train_dat.mat')
# X1_train = dat_train["X1"]
# X2_train = dat_train["X2"]
# X1err_train = dat_train["X1err"]
# X2err_train = dat_train["X2err"]
# X1nom_train = dat_train["X1nom"]
# X2nom_train = dat_train["X2nom"]
# U_train = dat_train["U"]
#
# # Load validation data from .mat file
# dat_val = scipy.io.loadmat('converted_val_dat.mat')
# X1_val = dat_val["X1"]
# X2_val = dat_val["X2"]
# X1err_val = dat_val["X1err"]
# X2err_val = dat_val["X2err"]
# X1nom_val = dat_val["X1nom"]
# X2nom_val = dat_val["X2nom"]
# U_val = dat_val["U"]

print('Generated Dataset')

## Load data from .npz file
# Load training data from .npz file
npz_train = np.load('train_dat.npz')
X1_train = npz_train["X1"]
X2_train = npz_train["X2"]
X1err_train = npz_train["X1err"]
X2err_train = npz_train["X2err"]
X1nom_train = npz_train["X1nom"]
X2nom_train = npz_train["X2nom"]
U_train = npz_train["U"]

npz_train.close()
npz_val = np.load('val_dat.npz')
X1_val = npz_val["X1"]
X2_val = npz_val["X2"]
X1err_val = npz_val["X1err"]
X2err_val = npz_val["X2err"]
X1nom_val = npz_val["X1nom"]
X2nom_val = npz_val["X2nom"]
U_val = npz_val["U"]

npz_val.close()

## Load Another dataset
# Uncomment the appropriate line depending on the dataset to load
# dat = scipy.io.loadmat('koopman_dat.mat')
# dat = scipy.io.loadmat('koopman_dat_err.mat')
# dat = scipy.io.loadmat('koopman_dat_err_wxyz.mat')

# # Extract training and validation data
# train_val_split_index = 5000

# X1_train = dat['data'][0:train_val_split_index-1, 1:23].T
# X2_train = dat['data'][1:train_val_split_index, 1:23].T
# X1nom_train = dat['data_nom'][0:train_val_split_index-1, 1:23].T
# X2nom_train = dat['data_nom'][1:train_val_split_index, 1:23].T
# X1err_train = dat['data_err'][0:train_val_split_index-1, 1:23].T
# X2err_train = dat['data_err'][1:train_val_split_index, 1:23].T
# U_train = dat['data'][0:train_val_split_index-1, 23:27].T

# X1_val = dat['data'][train_val_split_index:end-2, 1:23].T
# X2_val = dat['data'][train_val_split_index+1:end-1, 1:23].T
# X1nom_val = dat['data_nom'][train_val_split_index:end-2, 1:23].T
# X2nom_val = dat['data_nom'][train_val_split_index+1:end-1, 1:23].T
# X1err_val = dat['data_err'][train_val_split_index:end-2, 1:23].T
# X2err_val = dat['data_err'][train_val_split_index+1:end-1, 1:23].T
# U_val = dat['data'][train_val_split_index:end-2, 23:27].T

# Compute lengths of training and validation datasets
train_data_len = X1_train.shape[1]
val_data_len = X1_val.shape[1]

n_basis = 3
add_dim = 6

## Process training data based on the training method
if training_method == 'pure_data':
    Z1_train = np.zeros((nx + 9 * n_basis + add_dim, train_data_len))
    Z2_train = np.zeros((nx + 9 * n_basis + add_dim, train_data_len))

    for i in range(train_data_len):
        x1 = X1_train[:, i]
        x2 = X2_train[:, i]

        z1 = np.concatenate([x1[0:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), construct_basis(x1, n_basis)], axis=0)
        z2 = np.concatenate([x2[0:3].reshape(-1, 1), x2[3:6].reshape(-1, 1), construct_basis(x2, n_basis)], axis=0)
        print('x1[0:3].reshape:', x1[0:3].reshape(-1, 1).shape)
        print('x1[3:6].reshape:', x1[3:6].reshape(-1, 1).shape)
        print('construct_basis shape:', construct_basis(x1, n_basis).shape)
        print('z1 shape:', z1.ravel().shape)

        # print('z1 size', z1.ravel().shape)
        Z1_train[:, i] = z1.ravel()
        Z2_train[:, i] = z2.ravel()

elif training_method == 'hybrid1':
    Z1_train = np.zeros((nx + 9 * n_basis, train_data_len))
    Z2_train = np.zeros((nx + 9 * n_basis, train_data_len))
    Z1nom_train = np.zeros((nx + 9 * n_basis, train_data_len))
    Z2nom_train = np.zeros((nx + 9 * n_basis, train_data_len))

    for i in range(train_data_len):
        x1 = X1_train[:, i]
        x2 = X2_train[:, i]

        z1 = np.concatenate([x1[0:6].reshape(-1, 1), construct_basis(x1, n_basis)])
        z2 = np.concatenate([x2[0:6].reshape(-1, 1), construct_basis(x2, n_basis)])
        Z1_train[:, i] = z1.ravel()
        Z2_train[:, i] = z2.ravel()

        x1 = X1nom_train[:, i]
        x2 = X2nom_train[:, i]
        z1 = np.concatenate([x1[0:6].reshape(-1, 1), construct_basis(x1, n_basis)])
        z2 = np.concatenate([x2[0:6].reshape(-1, 1), construct_basis(x2, n_basis)])
        Z1nom_train[:, i] = z1.ravel()
        Z2nom_train[:, i] = z2.ravel()

elif training_method == 'hybrid2':
    Z1err_train = np.zeros((nx + 9 * n_basis, train_data_len))
    Z2err_train = np.zeros((nx + 9 * n_basis, train_data_len))

    for i in range(train_data_len):
        x1 = X1err_train[:, i]
        x2 = X2err_train[:, i]
        print('hi', construct_basis(x1, n_basis).shape)
        z1 = np.concatenate([x1[0:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), construct_basis(x1, n_basis)])
        z2 = np.concatenate([x2[0:3].reshape(-1, 1), x2[3:6].reshape(-1, 1), construct_basis(x2, n_basis)])
        Z1err_train[:, i] = z1.ravel()
        Z2err_train[:, i] = z2.ravel()

# Training via EDMD
if training_method == 'pure_data':
    A, B = EDMD(Z1_train, Z2_train, U_train)

elif training_method == 'hybrid1':
    A, B = EDMD(Z1_train, Z2_train, U_train)
    # print(A.shape)
    A_nom, B_nom = EDMD(Z1nom_train, Z2nom_train, U_train)  # On the nominal states

elif training_method == 'hybrid2':
    A_err, B_err = EDMD(Z1err_train, Z2err_train, U_train)  # On the state errors

C = np.zeros((nx + add_dim, nx + 9 * n_basis + add_dim))
C[0:nx + add_dim, 0:nx + add_dim] = np.eye(nx + add_dim)
# print(C.shape)


## Get prediction accuracy on training dataset
if training_method == 'pure_data':
    Z2_predicted = np.dot(A, Z1_train) + np.dot(B, U_train)

elif training_method == 'hybrid1':
    diff = np.dot(A, Z1_train) - np.dot(A_nom, Z1nom_train) + np.dot((B - B_nom), U_train)

elif training_method == 'hybrid2':
    Z2_error = np.dot(A_err, Z1err_train) + np.dot(B_err, U_train)

Xpred_train = np.zeros((nx + add_dim, train_data_len))
for i in range(train_data_len):
    if training_method == 'pure_data':
        Xpred_train[:, i] = np.dot(C, Z2_predicted[:, i])
    elif training_method == 'hybrid1':
        Xpred_train[:, i] = np.dot(C, diff[:, i]) + X2nom_train[:, i]
    elif training_method == 'hybrid2':
        Xpred_train[:, i] = np.dot(C, Z2_error[:, i]) + X2nom_train[:, i]

## Process validation dataset, to map states to basis vectors
if training_method == 'pure_data':
    Z1_val = np.zeros((nx + 9 * n_basis + add_dim, val_data_len))
    for i in range(val_data_len):
        x1 = X1_val[:, i]
        basis = construct_basis(x1, n_basis)
        z1 = np.concatenate([x1[:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), basis], axis=0)
        Z1_val[:, i] = z1.ravel()

elif training_method == 'hybrid1':
    Z1_val = np.zeros((nx + 9 * n_basis, val_data_len))
    Z1nom_val = np.zeros((nx + 9 * n_basis, val_data_len))
    for i in range(val_data_len):
        x1 = X1_val[:, i]
        basis = construct_basis(x1, n_basis)
        print('basis',basis.shape)
        z1 = np.concatenate([x1[:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), basis], axis=0)
        print(z1.shape)
        Z1_val[:, i] = z1.ravel()

        x1 = X1nom_val[:, i]
        basis = construct_basis(x1, n_basis)
        z1 = np.concatenate([x1[:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), basis], axis=0)
        Z1nom_val[:, i] = z1.ravel()

elif training_method == 'hybrid2':
    Z1err_val = np.zeros((nx + 9 * n_basis, val_data_len))
    for i in range(val_data_len):
        x1 = X1err_val[:, i]
        basis = construct_basis(x1, n_basis)
        z1 = np.concatenate([x1[:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), basis], axis=0)
        Z1err_val[:, i] = z1.ravel()

# Get prediction accuracy on validation dataset
if training_method == 'pure_data':
    Z2_predicted = np.dot(A, Z1_val) + np.dot(B, U_val)

elif training_method == 'hybrid1':
    diff = np.dot(A, Z1_val) - np.dot(A_nom, Z1nom_val) + np.dot((B - B_nom), U_val)

elif training_method == 'hybrid2':
    Z2_error = np.dot(A_err, Z1err_val) + np.dot(B_err, U_val)

Xpred_val = np.zeros((nx + add_dim, val_data_len))
for i in range(val_data_len):
    if training_method == 'pure_data':
        Xpred_val[:, i] = np.dot(C, Z2_predicted[:, i])

    elif training_method == 'hybrid1':
        Xpred_val[:, i] = np.dot(C, diff[:, i]) + X2nom_val[:, i]

    elif training_method == 'hybrid2':
        Xpred_val[:, i] = np.dot(C, Z2_error[:, i]) + X2nom_val[:, i]

## Get results
RMSE_labels = ["x", "dx", "quat_angle", "theta", "wb"]
RMSE_training = compute_rmse(Xpred_train, X2_train)
RMSE_validation = compute_rmse(Xpred_val, X2_val)

print("RMSE Training")
for label, value in zip(RMSE_labels, RMSE_training):
    print(f"{label:10}: {value:.4e}")

print('RMSE Validation')
for label, value in zip(RMSE_labels, RMSE_validation):
    print(f"{label:10}: {value:.4e}")