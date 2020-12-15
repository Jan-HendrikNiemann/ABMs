#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:00:19 2020

Generate measurements.

@author: Jan-Hendrik Niemann
"""


import os
import time as tm
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from VoterModel import markov_jump_process
from aux.auxiliary import random_init


def __jumpprocess(x_init, num_samples, gamma, gamma_prime, t_step, T_max):
    """
    Auxiliary function to parallelize the generation of trajectory data for
    the jump process

    Parameters
    ----------
    x_init : ndarray
        Initial population.
    num_samples : int
        Number of samples (repetions) generated from one initial values.
    gamma : ndarray
        Square array of size len(x_init).
    gamma_prime : ndarray
        Square array of size len(x_init).
    t_step : float
        Time step.
    T_max : int or float
        Maximum time horizon (also known as lag time).

    Returns
    -------
    x_trajectory : ndarray
        num_samples total trajectory of the system state for given time
        horizon and initial value. The order is
        [num_types, num_timesteps, num_samples, num_testpoints]

    """
    num_timesteps = int(np.round(T_max / t_step)) + 1
    num_types = x_init.shape[0]
    x_trajectory = np.empty([num_types, num_timesteps, num_samples])

    for j in range(num_samples):
        x_trajectory[:, :, j] = markov_jump_process(x_init,
                                                    gamma,
                                                    gamma_prime,
                                                    t_step,
                                                    T_max,
                                                    seed=None)

    return x_trajectory


def generate_data(num_types, num_agents, num_testpoints, num_samples, T_max,
                  t_step, gamma, gamma_prime):
    """
    Generates trajectory data using the jump process (JP)

    Parameters
    ----------
    num_types : int
        Number of different types.
    num_agents : int
        Number of agents.
    num_testpoints : int
        Number of training points.
    num_samples : int
        Number of repetitions per training point.
    T_max : float
        Maximum simulation time (also known as lag time).
    t_step : float
        RTime step.
    gamma : ndarray
        Transition rate constants (adaptive).
    gamma_prime : ndarray
        Transition rate constants (spontaneous).

    Returns
    -------
    x_trajectory : ndarray
        num_samples total trajectory of the system state for given time
        horizon and initial value. The order is
        [num_types, num_timesteps, num_samples, num_testpoints]
    x_init : ndarray
        Initial population.

    """

    x_init = np.empty([num_types, num_testpoints])

    num_cores = multiprocessing.cpu_count()

    for i in range(num_testpoints):
        x_init[:, i] = random_init(num_agents, num_types)
    x_trajectory = Parallel(n_jobs=num_cores, verbose=11)(delayed(__jumpprocess)(x_init[:, i],
                                                                                 num_samples,
                                                                                 gamma,
                                                                                 gamma_prime,
                                                                                 t_step,
                                                                                 T_max) for i in range(num_testpoints))
    x_trajectory = np.transpose(x_trajectory, (1, 2, 3, 0))

    return x_trajectory, x_init


# %% Settings

# Workspace directory
dir_path = 'data/'
dir_name = dir_path + 'raw'

# Lag time and time step
T_max = 0.01
t_step = 0.01

# Rate constants
gamma = np.array([[0, 2, 1],
                  [1, 0, 2],
                  [2, 1, 0]], dtype=float)
gamma_prime = 0.01 * (np.ones_like(gamma) - np.eye(len(gamma)))

num_agent_list = [10, 100, 1000]
num_samples_list = [10, 100, 1000]

num_trainingpoints_list = [7, 515, 10000]


# %% Create target directory

num_types = len(gamma)

overall_time = tm.time()

# Create directory
try:
    # Create target Directory
    os.mkdir(dir_name)
    print("Directory ", dir_name, "Created")
except FileExistsError:
    print("Directory ", dir_name, "already exists")


# %% Measurements and preparations for point-wise estimates

for i, num_agents in enumerate(num_agent_list):
    for j, num_samples in enumerate(num_samples_list):

        num_trainingpoints = num_trainingpoints_list[i]

        # Check if file already exists
        if os.path.isfile(dir_name + '/out_' + str(i) + '_' + str(j) + '.npz'):
            print('\nSimulation with %d agents, %d samples, %d trainingpoints already exists. Continue with next.\n' % (num_agents, num_samples, num_trainingpoints))
            continue
        else:
            print('\nSimulating %d agents, %d samples, %d trainingpoints\n' % (num_agents, num_samples, num_trainingpoints))

        # Print setting to file
        with open(dir_name + '/parameter_settings_' + str(i) + '_' + str(j) + '.txt', 'w') as file:
            file.write('- - - - Parameter settings - - - -\n\n')
            file.write('\nNumber of types: ' + str(num_types))
            file.write('\nGamma:\n' + str(gamma))
            file.write('\nGamma_prime:\n' + str(gamma_prime))
            file.write('\n\nTime step: ' + str(t_step))
            file.write('\nMaximum simulation time: ' + str(T_max))
            file.write('\nList of all agent numbers:\n' + str(num_agent_list))
            file.write('\nList of all sample numbers:\n' + str(num_samples_list))
            file.write('\n\n\n- - - - Current settings - - - -\n\n')
            file.write('\nNumber of agents: ' + str(num_agents))
            file.write('\nNumber of samples: ' + str(num_samples))
            file.write('\nNumber of trainingpoints: ' + str(num_trainingpoints))

        # Start clock
        start_time = tm.time()

        # Run simulation
        trajectory, x_init = generate_data(num_types,
                                           num_agents,
                                           num_trainingpoints,
                                           num_samples,
                                           T_max,
                                           t_step,
                                           gamma,
                                           gamma_prime)

        # Save result and parameters
        np.savez_compressed(dir_name + '/out_' + str(i) + '_' + str(j),
                            trajectory=trajectory,
                            x_init=x_init,
                            gamma=gamma,
                            gamma_prime=gamma_prime,
                            num_agents=num_agents,
                            num_samples=num_samples,
                            num_trainingpoints=num_trainingpoints,
                            t_step=t_step,
                            T_max=T_max)

        # End clock
        string = '{:.2f} seconds'.format(tm.time() - start_time)
        with open(dir_name + '/parameter_settings_' + str(i) + '_' + str(j) + '.txt', 'a') as file:
            file.write('\n\nElapsed time: ' + string)


# Total end clock in last file
string = '{:.2f} seconds'.format(tm.time() - overall_time)
with open(dir_name + '/parameter_settings_' + str(i) + '_' + str(j) + '.txt', 'a') as file:
    file.write('\n\nTotal elapsed time: ' + string)
