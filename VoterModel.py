#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:28:35 2020

Voter model defined on a complete network. Simulated using Gillespie's
stochastic simulation algorithm.

@author: Jan-Hendrik Niemann
"""

import numpy as np


def markov_jump_process(x_init, gamma, gamma_prime, t_step, T_max, seed=None):
    """
    Voter model implemented as Markov jump process

    Parameters
    ----------
    x_init : ndarray
        Initial population.
    gamma : ndarray
        Matrix containing transition rate constants for imitation
    gamma_prime : ndarray
        Matrix containing transition rate constants for exploration
    t_step : float
        Time step for output.
    T_max : int or float
        Time horizon.
    seed : int, optional
        Seed of random process. The default is None.

    Returns
    -------
    X : ndarray
        Trajectory of the system state for given time horizon.

    """
    if seed is not None:
        np.random.seed(seed)

    # Number of timesteps for saving
    num_timesteps = int(np.round(T_max / t_step)) + 1

    # Number of agents
    num_agents = np.sum(x_init)

    # Number of types
    num_types = x_init.shape[0]

    X = np.zeros([num_types, num_timesteps])
    alpha = np.zeros_like(gamma, dtype=float)

    # State
    x = np.zeros([num_types, 1])
    x[:, 0] = x_init

    # Time
    t = np.array([0])

    k = 0
    while t[k] < T_max:
        for i in range(num_types):
            for j in range(num_types):
                alpha[i, j] = (gamma[i, j] / num_agents * x[i, k] * x[j, k]
                               + gamma_prime[i, j] * x[i, k])
            sum_alpha = np.sum(alpha, axis=1)

        lmbda = np.sum(alpha)

        if lmbda == 0:
            break

        p = np.random.uniform(0, 1)

        # Determine time for the event to happen
        tau = 1 / lmbda * np.log(1 / p)

        t = np.hstack((t, np.array([t[k] + tau])))

        p = np.random.uniform(0, 1)
        i = 0
        while np.sum(sum_alpha[:i + 1]) / lmbda < p:
            i = i + 1

        j = 0
        if i == 0:
            while np.sum(alpha[i, :j + 1]) / lmbda < p:
                j = j + 1
        else:
            while np.sum(sum_alpha[:i]) / lmbda + np.sum(alpha[i, :j + 1]) / lmbda < p:
                j = j + 1

        x = np.hstack((x, x[:, k][:, None]))

        k = k + 1
        x[i, k] = x[i, k] - 1
        x[j, k] = x[j, k] + 1

    # Save for output
    for i in range(num_timesteps):
        idx = np.argmin(t <= i * t_step) - 1
        X[:, i] = x[:, idx]

    return X


# %%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    # Parameter setting
    num_agents = 25
    T_max = 10
    t_step = 0.01

    # Transition rate constants
    gamma = np.array([[0, 2, 1],
                      [1, 0, 2],
                      [2, 1, 0]],
                     dtype=float)
    gamma_prime = 0.01 * np.array([[0, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 0]],
                                  dtype=float)

    # Initial aggregate state
    x_init = np.array([0.2 * num_agents,
                       0.3 * num_agents,
                       0.5 * num_agents],
                      dtype=int)

    # Correct number of agents
    num_agents = np.sum(x_init)

    rate_constants = np.array([gamma, gamma_prime])
    num_types = gamma.shape[0]

    # Run simulation
    start = time.time()
    trajectory = markov_jump_process(x_init,
                                     gamma,
                                     gamma_prime,
                                     t_step,
                                     T_max,
                                     seed=None)
    stop = time.time()

    fig = plt.figure()
    for i in range(num_types):
        plt.step(np.linspace(0, T_max, trajectory.shape[1]),
                 trajectory[i, :],
                 label='$X_{' + str(i + 1) + '}(t)$')
    plt.legend(loc=1)
    plt.xlabel('Time $t$')
    plt.ylabel('Number of agents $X_i(t)$')
    plt.ylim([-0.05 * num_agents, 1.05 * num_agents])
    plt.show()

    print('\nElapsed time: %.4f seconds\n' % (stop - start))
