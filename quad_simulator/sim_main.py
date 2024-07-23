import timeit
import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate
from init_modules import instantiate_quadrotor, instantiate_controller, instantiate_trajectory
from sim_utils import data_plotting, get_metrics, extract_data

# Set simulation parameters
np.random.seed(1)
sampling_rate                       = 0.05  # Sampling rate in seconds
sim_duration, desired_speed, radius = 200, 1.0, 2.5
num_samples = sim_duration / sampling_rate
# print(num_samples)
# Instantiate trajectory planner
planner                     = instantiate_trajectory(radius, sim_duration, desired_speed)
#
# Instantiate quadrotor simulation model and initial state
quadrotor, initial_state    = instantiate_quadrotor(radius)

# Instantiate controller
controller                  = instantiate_controller()

# Run simulation
start_time  = timeit.default_timer()
(time_sim, states, control, ref, state_dot, rSpeeds, state_nom, state_err) = simulate(initial_state, quadrotor, controller, planner, sim_duration, sampling_rate)
# print('exit')
end_time    = timeit.default_timer()
print(f'Sim duration [s]: {end_time - start_time:.2f}s')

# Get results and metrics
pos_mse, vel_mse = get_metrics(states, ref, desired_speed, radius)

# Extract data
extract_data(time_sim, states, ref, control, state_nom, state_err)

# Plotting
data_plotting(time_sim, states, state_dot, ref, control)

plt.show()