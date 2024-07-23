import numpy as np


def simulate(initial_state, quadrotor, controller, traj_planner, t_final, t_step):
    N_sim       = int(t_final / t_step)
    num_states  = 13
    time_sim    = np.zeros(shape=(N_sim,))
    X_log       = np.zeros(shape=(N_sim, num_states))
    X_dot_log   = np.zeros(shape=(N_sim, num_states))
    U_log       = np.zeros(shape=(N_sim, 4))
    Xref_log    = np.zeros(shape=(N_sim, 17))
    rSpeeds_log = np.zeros(shape=(N_sim, 4))
    state_nom_log = np.zeros(shape=(N_sim, num_states))
    state_err_log = np.zeros(shape=(N_sim, num_states))

    state       = initial_state

    for i in range(N_sim):
        time    = (i+1)*t_step
        ref     = traj_planner.update(time)
        # print('i', i, 'dt/dt_sim', (int(controller.DT / t_step)), 'mod', i % (int(controller.DT / t_step)) == 0)
        # if i % (int(controller.DT / t_step)) == 0:
        #     control = controller.update(time, state, ref)
        #
        # for se3 controller
        control = controller.update(time, state, ref)

        # control = controller.update(time, state, ref)
        # controls_se3 = controller_se3.update(time, state, ref)
        # print('out of update')

        cmd_motor_speeds = control[0:4]
        # print('motor speeds out of update', cmd_motor_speeds)
        state, rotor_speeds, state_dot, state_nom, state_err = quadrotor.step(state, cmd_motor_speeds, t_step, time)
        # print('state after step', state)
        time_sim[i]         = time
        X_log[i, :]         = np.squeeze(np.array(state))
        X_dot_log[i, :]     = np.squeeze(np.array(state_dot))
        Xref_log[i, :]      = np.squeeze(np.array(ref))
        U_log[i, :]         = np.squeeze(np.array(cmd_motor_speeds))
        rSpeeds_log[i, :]   = np.squeeze(np.array(rotor_speeds))
        state_nom_log[i, :] = np.squeeze(np.array(state_nom))
        state_err_log[i, :] = np.squeeze(np.array(state_err))

        if i % (1/t_step) == 0:
            print('Sim time [s]:', i*t_step)

    return time_sim, X_log, U_log, Xref_log, X_dot_log, rSpeeds_log, state_nom_log, state_err_log
