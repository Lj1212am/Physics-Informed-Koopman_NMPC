from koopman.imports import *
from koopman.quad_dynamics import quad_dynamics


def generate_dataset(X0, n_control, t_span, flag):

    # Set true and nominal system params
    params = {
        "mass": 0.3,
        "J": np.diag([1.43e-5, 1.43e-5, 2.89e-5]),
        "g": 9.81
    }
    params_nom = {
        "mass": params["mass"] * 2,
        "J": np.diag([(1.43e-5) * 1.5, (1.43e-5) * 1.5, (2.89e-5) * 1.2]),
        "g": params["g"]
    }

    # Generate random inputs for U
    mu = np.array([0, 0, 0, 0])
    if flag == "train":
        # Sigma = np.diag([0.3, 5e-5, 5e-5, 5e-5])
        Sigma = np.diag([10, 10, 10, 10])
    elif flag == "val":
        # Sigma = np.diag([0.6, 5e-5, 5e-5, 5e-5])
        Sigma = np.diag([20, 20, 20, 20])
    else:
        raise ValueError("Invalid flag value. Expected 'train' or 'val'.")
    U_rnd = np.random.multivariate_normal(mu, Sigma, n_control)
    U_rnd[:, 0] += params["mass"] * params["g"]

    # Initialize matrices for storing trajectories
    traj_len = len(t_span) - 1
    X = np.zeros((len(X0), traj_len * n_control))
    X1 = np.zeros((len(X0), traj_len * n_control))
    X2 = np.zeros((len(X0), traj_len * n_control))
    X1err = np.zeros((len(X0), traj_len * n_control))
    X2err = np.zeros((len(X0), traj_len * n_control))
    X1nom = np.zeros((len(X0), traj_len * n_control))
    X2nom = np.zeros((len(X0), traj_len * n_control))
    U = np.zeros((4, traj_len * n_control))

    # Main loop to generate trajectories
    for i in range(n_control):
        u = U_rnd[i, :].T


        # Simulate true trajectory
        # x = np.array(odeint(lambda x, t: quad_dynamics(t, x, u, params), X0, t_span))
        x = np.array(odeint(lambda X, t: quad_dynamics(t, X, u, params), X0, t_span))

        # Generate one-step predictions using nominal model
        x_nominal = np.zeros((len(t_span), len(X0)))
        for k in range(len(t_span)):
            x_nominal[k, :] = np.array(odeint(lambda x, t: quad_dynamics(t, x, u, params_nom),  x[k, :], t_span[:2])[-1])

        # Organize the data
        X1[:, i*traj_len:(i+1)*traj_len] = x[:-1, :].T
        X2[:, i*traj_len:(i+1)*traj_len] = x[1:, :].T
        X1nom[:, i*traj_len:(i+1)*traj_len] = x_nominal[:-1, :].T
        X2nom[:, i*traj_len:(i+1)*traj_len] = x_nominal[1:, :].T
        X1err[:, i*traj_len:(i+1)*traj_len] = x[:-1, :].T - x_nominal[:-1, :].T
        X2err[:, i*traj_len:(i+1)*traj_len] = x[1:, :].T - x_nominal[1:, :].T
        u = u.reshape(4, 1)
        U[:, i*traj_len:(i+1)*traj_len] = np.tile(u, (1, traj_len))

    return X1, X2, X1err, X2err, X1nom, X2nom, U