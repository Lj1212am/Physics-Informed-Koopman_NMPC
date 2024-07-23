from quad_simulator.cts_quadrotor import QuadrotorDynamics
from quad_simulator.TrajectoryPlanners.waypoint_traj import WaypointTraj
from quad_simulator.Controllers.se3_control import SE3Control
from quad_simulator.Controllers.nmpc_controller import QuadrotorNMPC
from quad_simulator.Controllers.koopman_nmpc_controller import QuadrotorKoopNMPC
from quad_simulator.crazyflie_params import quad_params
import numpy as np


def instantiate_quadrotor(length):
    quadrotor = QuadrotorDynamics(quad_params)
    quadrotor.state_noise = np.random.normal(size=13, scale=1e-3)

    # Initialize position and velocity
    pos0 = np.array([length, 0, 0])
    vel0 = np.array([0, 1, 0])

    initial_state = np.hstack((pos0, vel0, np.array([0, 0., 0., 1.]), np.zeros(3, )))  # [i,j,k,w]
    return quadrotor, initial_state


## Updated to do figure 8s
def instantiate_trajectory(length, t_final, desired_traj_speed):
    radius = length
    t_plot = np.linspace(0, t_final, num=500)
    #
    # # Parametric equations for figure-8
    # x_traj = (radius * np.cos(t_plot)) / (1 + np.sin(t_plot)**2)
    # y_traj = (radius * np.cos(t_plot) * np.sin(t_plot)) / (1 + np.sin(t_plot)**2)

    # Circle Traj
    x_traj = radius * np.cos(t_plot)
    y_traj = radius * np.sin(t_plot)

    z_traj = np.zeros((len(t_plot),))

    points = np.stack((x_traj, y_traj, z_traj), axis=1)
    trajectory = WaypointTraj(points, desired_traj_speed)
    return trajectory


def instantiate_controller():
    # controller = QuadrotorNMPC()
    controller      = SE3Control()

    return controller


def instantiate_controller_koop(results, training_method):
    controller = QuadrotorKoopNMPC()
    if training_method == 'pure_data':
        controller.set_koopman(training_method, A=results['A'], B=results['B'], C=results['C'])
    elif training_method == 'hybrid1':
        controller.set_koopman(training_method, A=results['A'], B=results['B'], C=results['C'], A_nom=results['A_nom'],
                               B_nom=results['B_nom'])
    elif training_method == 'hybrid2':
        # Assuming hybrid2 uses A_err and B_err instead of A and B
        # Adjust based on actual implementation needs
        controller.set_koopman(training_method, A_err=results['A_err'], B_err=results['B_err'], C=results['C'])

    return controller
