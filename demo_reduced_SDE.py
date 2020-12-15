#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:33:57 2020

Demo how to simulate the reduced data-driven SDE.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import d3s.observables as observables


def data_sde(L, p, x_init, T_max, t_step, seed=None):
    """
    Euler-Maruyama scheme to simulate the reduced SDE

    Parameters
    ----------
    L : ndarray
        Koopman generator approximation matrix.
    p : int
        Degree of highest-order monomial.
    x_init : ndarray
        Initial state.
    T_max : float
        Time horizon for simulation.
    t_step : float
        Time step size.
    seed : int, optional
        Seed of random process. The default is None.

    Returns
    -------
    C : ndarray
        Trajectory of the SDE simulated for interval [0, T_max].

    """

    if seed is not None:
        np.random.seed(seed)

    num_types = len(x_init)

    # Define observables
    psi = observables.monomials(p)

    # Number of timestep to be simulated
    num_timesteps = int(np.round(T_max / t_step)) + 1

    C = np.zeros([num_types - 1, num_timesteps])
    C[:, 0] = x_init[0:num_types - 1]

    for k in range(num_timesteps - 1):
        # Get random number
        Wt = np.random.normal(0, 1, num_types - 1)

        # Evaluate drift and diffusion in X given the data-driven L
        X = C[:, k]
        X = np.expand_dims(X, axis=1)
        Psi_c = psi(X)

        # Calculate drift
        b_c = L[:, 1:num_types].T @ Psi_c
        b_c = np.squeeze(b_c)

        # Calculate diffusion
        a_c_11 = L[:, 3].T @ Psi_c - 2 * b_c[0] * X[0, :]
        a_c_12 = L[:, 4].T @ Psi_c - b_c[0] * X[1, :] - b_c[1] * X[0, :]
        a_c_22 = L[:, 5].T @ Psi_c - 2 * b_c[1] * X[1, :]

        A = np.empty([num_types - 1, num_types - 1])
        A[0, 0] = a_c_11
        A[1, 0] = a_c_12
        A[0, 1] = a_c_12
        A[1, 1] = a_c_22

        # The matrix A can have eigenvalues that are numerically (close to)
        # zero. If the Cholesky decomposition cannot be computed, the diffusion
        # term sigma will not be updated in this step
        try:
            __ = sigma.shape
        except NameError:
            sigma = np.zeros_like(A)
        try:
            sigma = sp.linalg.cholesky(A, lower=True)
        except Exception as excp:
            pass

        C[:, k + 1] = C[:, k] + t_step * b_c + np.sqrt(t_step) * sigma @ Wt

    # Calculate missing last entrie
    C = np.vstack((C, np.zeros(num_timesteps)))
    C[2, :] = 1 - C[0, :] - C[1, :]

    return C


# %%

if __name__ == '__main__':

    # Parameters for the SDE simulation
    T_max = 25
    t_step = 0.01

    # Initial state
    x_init = np.array([0.2, 0.7, 0.1])

    with np.load('data/processed/matrix_2_2.npz') as data:
        print('Available in this data set:', data.files)
        L = data['L']
        num_types = data['num_types']
        degree = data['degree']

    # Number of timestep to be simulated
    num_timesteps = int(np.round(T_max / t_step)) + 1

    # Simulate SDE
    C = data_sde(L, degree, x_init, T_max, t_step)

    # Plot
    fig = plt.figure()
    for i in range(num_types):
        plt.plot(np.linspace(0, T_max, num_timesteps),
                 C[i, :],
                 label='$X_{' + str(i + 1) + '}(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel('Number of agents $X_i(t)$')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc=1)
    plt.show()
