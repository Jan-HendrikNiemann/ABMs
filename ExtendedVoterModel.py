#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:00:15 2020

Extended voter model defined on a network.

@author: Jan-Hendrik Niemann
"""


import numpy as np
from scipy.linalg import expm


def __transition_propabability_matrix(x, rate_constants, t_step):
    """
    Transition probability matrix. This function can be modified to work with
    different transition rules. Currently imitation and mutation only

    Parameters
    ----------
    x : ndarray
        Distribution or aggregate state.
    rate_constants : ndarray
        3 dim. tensor of rate constants. First dimension coincides with the
        number of transition rules.
    t_step : float
        Time step.

    Returns
    -------
    P : ndarray
        Transition probability matrix.

    """

    # Detect whether x is a distribution (connected networks only, no single agents)
    if np.sum(x) > 1:
        N = np.sum(x)
    else:
        N = 1

    # Modifiy the following two lines if more than two transitions are possible
    gamma = rate_constants[0, :, :]
    gamma_prime = rate_constants[1, :, :]

    P = 1 / N * np.outer(np.ones_like(x), x)
    P = np.multiply(P, gamma) + gamma_prime
    P = P - np.diag(np.sum(P, axis=1))
    P = expm(t_step * P)

    return P


def get_distribution(state, num_types, neighbors=None):
    """
    Distribution of types for full network

    Parameters
    ----------
    state : ndarray
        Full state vector for all agents.
    num_types : ndarray
        Number of types/opinions/traits/...
    neighbors : ndarray, optional
        If ndarray, distribution regarding neighbors is returned. The default
        is None.

    Returns
    -------
    freq : ndarray
        Frequencies of each types.

    """
    freq = np.zeros(num_types)

    if neighbors is None:
        num_agents = len(state)
        val, counts = np.unique(state, return_counts=True)

    else:
        num_agents = len(state[neighbors])
        val, counts = np.unique(state[neighbors], return_counts=True)

    val.astype(int)

    for i, idx in enumerate(val.astype(int)):
        freq[idx] = counts[i]

    freq = freq / num_agents

    return freq


def get_neighborhood(i, network, r=1):
    """
    Returns vector with indices of neighbors including self

    Parameters
    ----------
    i : int
        Agent number i.
    network : ndarray
        Adjacency matrix. Note that diag(network) = 1 is assumed (and needed/set)
    r : int, optional
        Radius of view, i.e., degree of neighborhood. First order neighbors are
        direct neighbors, second order neighbors are neighbors of neighbors.
        The default is 1.

    Returns
    -------
    neighbors : ndarray
        Vector with indices of neighbors including self.

    """

    neighbors = np.argwhere(network[int(i), :] == 1)
    neighbors = np.squeeze(neighbors)

    # Get augmented neighbors
    if r > 1:
        augmented_neighbors = np.zeros([])
        for each in neighbors:
            augmented_neighbors = np.hstack((augmented_neighbors,
                                             get_neighborhood(each,
                                                              network,
                                                              r - 1)))

        neighbors = np.hstack((neighbors, augmented_neighbors))
        neighbors = np.unique(neighbors)

    return neighbors


def __update_agent_state(current_agent_state, transition_probability, rng):
    """
    Get agent state for next time step

    Parameters
    ----------
    current_agent_state : int
        Current agent state.
    transition_probability : ndarray
        Transition probability vector corresponding to current state.
    rng : numpy random number generator

    Returns
    -------
    agent_state : int
        Agent state at next time step.

    """
    choice = rng.uniform()

    agent_state = current_agent_state

    for i in range(len(transition_probability)):
        if choice < sum(transition_probability[0: i + 1]):
            agent_state = i
            break

    return agent_state


def step(current_state, network, t_step, rate_constants, rng, r=1):
    """
    Returns (full)-state vector propagated with time step

    Parameters
    ----------
    current_state : ndarray
        Current (full)-state vector.
    network : ndarray
        Adjacency matrix. Note that diag(network) = 1 is assumed (and needed)
    t_step : float
        Time step.
    rate_constants : ndarray
        3 dim. tensor of rate constants. First dimension coincides with the
        number of transition rules.
    rng : numpy random number generator
    r : int, optional
        Radius of view, i.e., degree of neighborhood. First order neighbors are
        direct neighbors, second order neighbors are neighbors of neighbors.
        The default is 1.

    Returns
    -------
    new_state : ndarray
        Propagated (full)-state vector.

    """
    num_agents = len(current_state)

    num_types = rate_constants.shape[1]

    idx = np.arange(0, num_agents)
    # np.random.shuffle(idx)
    rng.shuffle(idx)

    new_state = np.copy(current_state)

    for i in idx:
        # Get neighbors
        neighbors = get_neighborhood(i, network, r)

        # Get local distribution
        loc_distr = get_distribution(state=current_state,
                                     num_types=num_types,
                                     neighbors=neighbors)

        # Get transition probability matrix.
        # Can be modified if more than two transitions are possible
        transition_probability = __transition_propabability_matrix(loc_distr,
                                                                   rate_constants,
                                                                   t_step)
        # Get current agent state
        current_agent_state = int(current_state[i])
        # Get agent state for the next time step
        new_agent_state = __update_agent_state(current_agent_state,
                                               transition_probability[current_agent_state, :],
                                               rng)
        # Set new agent state in (full)-state vector
        new_state[i] = new_agent_state

    return new_state


def sim_ABM(x_init, rate_constants, t_step, T_max, network=None, horizon=1, rng=None):
    """
    Simulate an agent-based model

    Parameters
    ----------
    x_init : ndarray
        Initial state for each agent.
    rate_constants : ndarray
        3 dim. tensor of rate constants. First dimension coincides with the
        number of transition rules.
    t_step : float
        Time step.
    T_max : float
        Maximum simulation time.
    network : ndarray, optional
        Adjacency matrix. Note that diag(network) = 1 is assumed (and needed).
        The default is None.
    horizon : int, optional
        Radius of view, i.e., degree of neighborhood. First order neighbors are
        direct neighbors, second order neighbors are neighbors of neighbors.
        The default is 1.
    rng : numpy random number generator
        The default is None

    Returns
    -------
    aggregate_state : ndarray
        Aggregated state of the temporal evolution.
    full_state : ndarray
        Temporal evolution for each agent.
    timeline : ndarray
        Vector of time steps (useful for easy plotting).

    """
    num_agents = len(x_init)
    num_types = rate_constants.shape[1]

    # Random number generator
    if rng is None:
        rng = np.random.default_rng()

    if network is None:
        print('Complete network used')
        network = np.ones([num_agents, num_agents])

    timeline = np.arange(0, T_max + t_step, t_step)

    full_state = np.empty([num_agents, len(timeline)])
    full_state[:, 0] = x_init

    aggregate_state = np.empty([num_types, len(timeline)])
    aggregate_state[:, 0] = get_distribution(x_init, num_types)

    for idx in range(len(timeline) - 1):
        full_state[:, idx + 1] = step(current_state=full_state[:, idx],
                                      network=network,
                                      t_step=t_step,
                                      rate_constants=rate_constants,
                                      rng=rng,
                                      r=horizon)
        aggregate_state[:, idx + 1] = get_distribution(full_state[:, idx + 1],
                                                       num_types)

    return aggregate_state, full_state, timeline


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
    distr = np.array([0.2 * num_agents,
                      0.3 * num_agents,
                      0.5 * num_agents],
                     dtype=int)

    # Correct number of agents
    num_agents = np.sum(distr)

    rate_constants = np.array([gamma, gamma_prime])
    num_types = gamma.shape[0]

    # Create random adjacency matrix
    network = np.random.uniform(size=[num_agents, num_agents])
    network[network < 0.5] = 0
    network[network != 0] = 1
    np.fill_diagonal(network, 1)
    network = network.astype(int)

    # Initialize ABM state
    x_init = np.zeros([0], dtype=int)
    for i in range(num_types):
        x_init = np.hstack((x_init, i * np.ones(distr[i], dtype=int)))
    np.random.shuffle(x_init)

    # Run simulation
    start = time.time()
    aggregate_state, full_state, timeline = sim_ABM(x_init,
                                                    rate_constants,
                                                    t_step,
                                                    T_max,
                                                    network)
    stop = time.time()

    fig = plt.figure()
    for i in range(num_types):
        plt.step(timeline,
                 aggregate_state[i, :] * num_agents,
                 label='$X_{' + str(i + 1) + '}(t)$')
    plt.legend(loc=1)
    plt.xlabel('Time $t$')
    plt.ylabel('Number of agents $X_i(t)$')
    plt.ylim([-0.05 * num_agents, 1.05 * num_agents])
    plt.show()

    fig = plt.figure()
    plt.imshow(network, cmap='binary')
    plt.title('Adjacency matrix')
    plt.xlabel('Index of agent')
    plt.ylabel('Index of agent')

    print('\nElapsed time: %.4f seconds\n' % (stop - start))
