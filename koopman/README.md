# Physics-Informed Koopman Operator (PIKO)
In this project, we implement a specialized variant of the Koopman operator, which we call Physics-Informed Koopman Operator (PIKO).
In this variant, we consider the case where we have some prior knowledge of the dynamics of the underlying system.
We then incorporate the knowledge through a careful selection of the observable functions that form the basis of the Koopman operator.

### Instructions
Main script: `main.m`

### File descriptions
`main.m`: There are 3 options available for the construction of the Koopman operator, `'pure_data'`, `'hybrid1'`, `'hybrid2'`.
The method `'pure_data'` is a vanilla version that does not leverage prior knowledge of the system dynamics.
The methods `'hybrid1'` and `'hybrid2'` are 2 approaches in which prior knowledge can be incorporated.
To select the option to use, use variable training_method.

`generate_dataset.m`: Creates the training and validation dataset based on the options given in main.m.  
To reduce execution time, these have been pre-generated and saved in `'train_dat_wquat.mat'` and `'val_dat_wquat.mat'`.
*Note:* Currently, the code to generate the datasets in `main.m` has been commented out, but should be activated during testing.

`quad_dynamics.m`: Formulate the dynamics for a quadrotor system.

`construct_basis.m`: Construct the observable/basis functions with the given dataset.

`EDMD.m`: Performs the operations required in Extended Dynamic Mode Decomposition (EDMD).

`compute_rmse.m`: Compute the root mean square errors (RMSEs) based on the one-step predictions and ground truth.

`results_to_compare.xlsx`: Set of results for the 3 methods at initial commit.

### About
For questions or bugs, please contact KongYao Chee (ckongyao@seas.upenn.edu).
