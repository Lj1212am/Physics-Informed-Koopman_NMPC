import timeit
import matplotlib.pyplot as plt

from quad_simulator.simulate import simulate
from quad_simulator.init_modules import instantiate_quadrotor, instantiate_controller_koop, instantiate_trajectory
from quad_simulator.sim_utils import data_plotting, get_metrics, extract_data

from koopman.imports import *
from koopman.EDMD import EDMD
from koopman.construct_basis import construct_basis
from koopman.compute_rmse import compute_rmse
from koopman.generate_dataset import generate_dataset
from scipy.spatial.transform import Rotation as R
from koopman.generate_dataset import generate_dataset


def trainEDMD(training_method='pure_data'):
    dt = 1e-3
    t_final = 0.1
    t_span = np.arange(0, t_final + dt, dt)

    results = {}

    # Initial state
    p0 = np.array([0, 0, 0])
    v0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    w0 = np.array([0.1, 0, 0])
    # Convert rotation matrix to quaternion using pyquaternion's Quaternion
    q = Quaternion(matrix=R0.T).elements
    x0 = np.concatenate([p0, v0, R0.ravel(), w0, q])
    nx = len(x0)

    ## Generate new data
    # # Training dataset
    # print("Generating Dataset...")
    # n_control = 100
    # X1_train, X2_train, X1err_train, X2err_train, X1nom_train, X2nom_train, U_train = \
    #     generate_dataset(x0, n_control, t_span, 'train')
    # # Here we initialize 3D arrays to store rotation matrices and then fill them with reshaped data
    # R1_train = np.zeros((3, 3, X1_train.shape[1]))
    # R2_train = np.zeros((3, 3, X2_train.shape[1]))
    # R1nom_train = np.zeros((3, 3, X1nom_train.shape[1]))
    # R2nom_train = np.zeros((3, 3, X2nom_train.shape[1]))
    #
    # for i in range(X1_train.shape[1]):
    #     R1_train[:, :, i] = X1_train[6:15, i].reshape(3, 3)
    #     R2_train[:, :, i] = X2_train[6:15, i].reshape(3, 3)
    #     R1nom_train[:, :, i] = X1nom_train[6:15, i].reshape(3, 3)
    #     R2nom_train[:, :, i] = X2nom_train[6:15, i].reshape(3, 3)
    # # Using pyquaternion to convert the rotation matrices to quaternions
    # # Q1_train = [Quaternion(matrix=R1_train[:, :, i]).elements for i in range(R1_train.shape[2])]
    # # Q2_train = [Quaternion(matrix=R2_train[:, :, i]).elements for i in range(R2_train.shape[2])]
    # # Q1nom_train = [Quaternion(matrix=R1nom_train[:, :, i]).elements for i in range(R1nom_train.shape[2])]
    # # Q2nom_train = [Quaternion(matrix=R2nom_train[:, :, i]).elements for i in range(R2nom_train.shape[2])]
    #
    # # Validation dataset
    # n_control_val = 50
    # X1_val, X2_val, X1err_val, X2err_val, X1nom_val, X2nom_val, U_val = \
    #     generate_dataset(x0, n_control_val, t_span, 'val')
    # # Create dictionary for training data
    # train_dat = {
    #     "X1": X1_train,
    #     "X2": X2_train,
    #     "X1err": X1err_train,
    #     "X2err": X2err_train,
    #     "X1nom": X1nom_train,
    #     "X2nom": X2nom_train,
    #     "U": U_train
    # }
    #
    # # Create dictionary for validation data
    # val_dat = {
    #     "X1": X1_val,
    #     "X2": X2_val,
    #     "X1err": X1err_val,
    #     "X2err": X2err_val,
    #     "X1nom": X1nom_val,
    #     "X2nom": X2nom_val,
    #     "U": U_val
    # }
    #
    # # Save the dictionaries to .npz files
    # np.savez('train_dat_train.npz', **train_dat)
    # np.savez('val_dat.npz', **val_dat)
    #
    # print("Generated dataset")

    # ## Generate new data
    # # Training dataset
    # print("Generating Dataset...")
    # n_control = 100
    # X1_train, X2_train, X1err_train, X2err_train, X1nom_train, X2nom_train, U_train = \
    #     generate_dataset(x0, n_control, t_span, 'train')
    # # Here we initialize 3D arrays to store rotation matrices and then fill them with reshaped data
    # R1_train = np.zeros((3, 3, X1_train.shape[1]))
    # R2_train = np.zeros((3, 3, X2_train.shape[1]))
    # R1nom_train = np.zeros((3, 3, X1nom_train.shape[1]))
    # R2nom_train = np.zeros((3, 3, X2nom_train.shape[1]))
    #
    # for i in range(X1_train.shape[1]):
    #     R1_train[:, :, i] = X1_train[6:15, i].reshape(3, 3)
    #     R2_train[:, :, i] = X2_train[6:15, i].reshape(3, 3)
    #     R1nom_train[:, :, i] = X1nom_train[6:15, i].reshape(3, 3)
    #     R2nom_train[:, :, i] = X2nom_train[6:15, i].reshape(3, 3)
    # # Using pyquaternion to convert the rotation matrices to quaternions
    # # Q1_train = [Quaternion(matrix=R1_train[:, :, i]).elements for i in range(R1_train.shape[2])]
    # # Q2_train = [Quaternion(matrix=R2_train[:, :, i]).elements for i in range(R2_train.shape[2])]
    # # Q1nom_train = [Quaternion(matrix=R1nom_train[:, :, i]).elements for i in range(R1nom_train.shape[2])]
    # # Q2nom_train = [Quaternion(matrix=R2nom_train[:, :, i]).elements for i in range(R2nom_train.shape[2])]
    #
    # # Validation dataset
    # n_control_val = 50
    # X1_val, X2_val, X1err_val, X2err_val, X1nom_val, X2nom_val, U_val = \
    #     generate_dataset(x0, n_control_val, t_span, 'val')
    # # Create dictionary for training data
    # train_dat = {
    #     "X1": X1_train,
    #     "X2": X2_train,
    #     "X1err": X1err_train,
    #     "X2err": X2err_train,
    #     "X1nom": X1nom_train,
    #     "X2nom": X2nom_train,
    #     "U": U_train
    # }
    #
    # # Create dictionary for validation data
    # val_dat = {
    #     "X1": X1_val,
    #     "X2": X2_val,
    #     "X1err": X1err_val,
    #     "X2err": X2err_val,
    #     "X1nom": X1nom_val,
    #     "X2nom": X2nom_val,
    #     "U": U_val
    # }
    #
    # # Save the dictionaries to .npz files
    # np.savez('train_dat.npz', **train_dat)
    # np.savez('val_dat.npz', **val_dat)
    #
    # print("Generated dataset")


    # Load training data from .npz file
    # npz_train = np.load('train_dat.npz')
    npz_train = np.load('koopman/train_data.npz')
    X1_train = npz_train["X1"]
    X2_train = npz_train["X2"]
    X1err_train = npz_train["X1err"]
    X2err_train = npz_train["X2err"]
    X1nom_train = npz_train["X1nom"]
    X2nom_train = npz_train["X2nom"]
    U_train = npz_train["U"]

    # npz_val = np.load('val_dat.npz')
    npz_val = np.load('koopman/validation_data.npz')
    X1_val = npz_val["X1"]
    X2_val = npz_val["X2"]
    X1err_val = npz_val["X1err"]
    X2err_val = npz_val["X2err"]
    X1nom_val = npz_val["X1nom"]
    X2nom_val = npz_val["X2nom"]
    U_val = npz_val["U"]

    npz_val.close()
    print("loaded dataset")
    # Compute lengths of training and validation datasets
    train_data_len = X1_train.shape[1]
    val_data_len = X1_val.shape[1]

    n_basis = 3
    add_dim = 0

    ## Process training data based on the training method
    if training_method == 'pure_data':
        Z1_train = np.zeros((nx + 9 * n_basis + add_dim, train_data_len))
        Z2_train = np.zeros((nx + 9 * n_basis + add_dim, train_data_len))

        for i in range(train_data_len):
            x1 = X1_train[:, i]
            x2 = X2_train[:, i]

            z1 = np.concatenate([x1[0:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), construct_basis(x1, n_basis)], axis=0)
            z2 = np.concatenate([x2[0:3].reshape(-1, 1), x2[3:6].reshape(-1, 1), construct_basis(x2, n_basis)], axis=0)
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

            z1 = np.concatenate([x1[0:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), construct_basis(x1, n_basis)])
            z2 = np.concatenate([x2[0:3].reshape(-1, 1), x2[3:6].reshape(-1, 1), construct_basis(x2, n_basis)])
            Z1err_train[:, i] = z1.ravel()
            Z2err_train[:, i] = z2.ravel()

    C = np.zeros((nx + add_dim, nx + 9 * n_basis + add_dim))
    C[0:nx + add_dim, 0:nx + add_dim] = np.eye(nx + add_dim)
    # print(C.shape)

    if training_method == 'pure_data':
        A, B = EDMD(Z1_train, Z2_train, U_train)
        results['A'] = A
        results['B'] = B
        results['C'] = C

    elif training_method == 'hybrid1':
        A, B = EDMD(Z1_train, Z2_train, U_train)
        A_nom, B_nom = EDMD(Z1nom_train, Z2nom_train, U_train)  # On the nominal states
        results['A'] = A
        results['B'] = B
        results['A_nom'] = A_nom
        results['B_nom'] = B_nom
        results['C'] = C

    elif training_method == 'hybrid2':
        A_err, B_err = EDMD(Z1err_train, Z2err_train, U_train)  # On the state errors
        results['A_err'] = A_err
        results['B_err'] = B_err
        results['C'] = C

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
            # print('basis', basis.shape)
            z1 = np.concatenate([x1[:3].reshape(-1, 1), x1[3:6].reshape(-1, 1), basis], axis=0)
            # print(z1.shape)
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

    return results


if __name__ == "__main__":
    # Set random seed
    np.random.seed(1)

    # EDMD on last flights data
    training_method = 'hybrid1'  # Options: 'pure_data', 'hybrid1', 'hybrid2'

    results = trainEDMD(training_method)

    # np.random.seed(2)
    sampling_rate = 0.05  # Sampling rate in seconds
    sim_duration, desired_speed, radius = 4, 1.0, 2.5
    num_samples = sim_duration / sampling_rate
    # print(num_samples)
    # Instantiate trajectory planner
    planner = instantiate_trajectory(radius, sim_duration, desired_speed)
    #
    # Instantiate quadrotor simulation model and initial state
    quadrotor, initial_state = instantiate_quadrotor(radius)

    # Instantiate controller
    controller = instantiate_controller_koop(results, training_method)

    # Run simulation
    start_time = timeit.default_timer()
    (time_sim, states, control, ref, state_dot, rSpeeds, state_nom, state_err) = simulate(initial_state, quadrotor,
                                                                                          controller, planner,
                                                                                          sim_duration, sampling_rate)
    # print('exit')
    end_time = timeit.default_timer()
    print(f'Sim duration [s]: {end_time - start_time:.2f}s')

    # Get results and metrics
    pos_mse, vel_mse = get_metrics(states, ref, desired_speed, radius)

    # Extract data
    # extract_data(time_sim, states, ref, control, state_nom, state_err)

    # Plotting
    data_plotting(time_sim, states, state_dot, ref, control)

    plt.show()
