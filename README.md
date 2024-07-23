
# Physics-Informed Koopman NMPC for Quadrotor Simulator



## Overview

This project looks to use the Koopman Operator with physics-informed data to develop a NMPC for a quadrotor simulator. There are currently issues with the integration of the Koopman Operator in the NMPC controller
## Features

- **Physics-Informed Koopman NMPC Controller**: Controller combining physics informed model of Quadrotor with Koopman Operator
- **Real-Time Simulation**: Provides a real-time simulation environment for a quadrotor.
- **Visualization Tools**: Integrated tools for visualizing the flight path and control actions of the quadrotor.



## Usage

To run the simulator, execute the following command:

```bash
python main_koop_sim.py
```

To run with only the NMPC controller run sim_main.py in the quad_simulator package.

To run only the Koopman Operator, run main.py in the koopman package.

If you would like to generate data uncomment the lines 22-86 in main_koop_sim.py and run the file. Make sure to change the locations that the data is loaded from.

If you want to use data derived from running the simulation:
1. Run the sim_main.py in the quad_simulator package. 
2. Then run the data_conversion.py file.
3. Finally, run the main_koop_sim.py file.

To change the Controller used from nmpc to SE3 for the sim_main.py file, Change the controller instantiated in the instantiate_controllers function in init_modules.py 


## Configuration

Adjust the training method in main_koop_sim.py on line 231.
Choose between 'pure_data', 'hybrid1', 'hybrid2'.



## Debugging

Currently the project is stuck at the integration of the Koopman Operator in the NMPC controller.

Here are the current avenues of debugging, that I have gone down:
* Matching quad dynamics with the quad_simulator  dynamics in cts_quadrotor.py
* Instead of Ipopt nmpc controller using QP MPC controller
* Testing from data generated, or data derived from the simulator.
* Ensuring psi functionality compared to equivalent numpy construct_basis.py function
* Trying to match RMSE for Koopman validation to the RMSE of the data of this project: https://github.com/sriram-2502/KoopmanMPC_Quadrotor
