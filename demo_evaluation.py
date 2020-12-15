#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:01:21 2020

Evalution of post processed data.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import d3s.observables as observables
from aux.auxiliary import RMSE, random_init_2


def drift(C, num_types, gamma, gamma_prime):
    """
    Drift term.

    Parameters
    ----------
    C : ndarray
        Initial state.
    num_types : int
        Number of types (dimension).
    gamma : ndarray
        Transition rate constants.
    gamma_prime : ndarray
        Transition rate constants.

    Returns
    -------
    b_C : ndarray
        Drift term evaluated in C.

    """

    Id = np.eye(num_types)
    b_C = np.zeros(num_types)

    for i in range(num_types):
        for j in range(num_types):
            nu_k = Id[:, j] - Id[:, i]
            b_C = b_C + (gamma[i, j] * C[i] * C[j] + gamma_prime[i, j] * C[i]) * nu_k

    return b_C


def diffusion_KM(C, num_types, num_agents, gamma, gamma_prime):
    """
    Diffusion term sigma*sigma^T.

    Parameters
    ----------
    C : ndarray
        Initial state.
    num_types : int
        Number of types (dimension).
    num_agents : int
        Number of agents.
    gamma : ndarray
        Transition rate constants.
    gamma_prime : ndarray
        Transition rate constants.

    Returns
    -------
    a_C : ndarray
        Diffusion term evaluated in C.

    """

    Id = np.eye(num_types)
    a_C = np.zeros([num_types, num_types])

    for i in range(num_types):
        for j in range(num_types):
            nu_k = Id[:, j] - Id[:, i]
            a_C = a_C + (gamma[i, j] * C[i] * C[j] + gamma_prime[i, j] * C[i]) / num_agents * np.outer(nu_k, nu_k)

    return a_C


# %% Settings

# Preprocessed data
data_path = 'data/processed/matrix_'

# Number of types
num_types = 3

num_agent_list = [10, 100, 1000]
num_sample_list = [10, 100, 1000]

# Approx. 10 % of the total available points in the discrete space but at most 10000
num_trainingpoints_list = [7, 515, 10000]

# Maximum degree of monomials
degree = 3

# Number of test points for evaluation
num_testpoints = 10000

# Transition rate constants
gamma = np.array([[0, 2, 1],
                  [1, 0, 2],
                  [2, 1, 0]], dtype=float)

# Transition rate constants
gamma_prime = 0.01 * np.array([[0, 1, 1],
                               [1, 0, 1],
                               [1, 1, 0]], dtype=float)


# %% Evaluate

RMSE_drift_coeff = np.zeros([len(num_agent_list), len(num_sample_list)], dtype=float)
RMSE_diffusion_coeff = np.zeros_like(RMSE_drift_coeff)
RMSE_drift_eval = np.zeros_like(RMSE_drift_coeff)
RMSE_diffusion_eval = np.zeros_like(RMSE_drift_coeff)

RMSE_eig = np.zeros_like(RMSE_drift_coeff)
MAE_eig = np.zeros_like(RMSE_drift_coeff)

# For all agents
for i in range(len(num_agent_list)):
    num_agents = num_agent_list[i]

    # Get new x_init for evaluation
    x_init = np.zeros([num_types, num_testpoints])

    for each in range(num_testpoints):
        x_init[:, each] = random_init_2(num_types)

    Y_theo = np.empty([num_types, num_testpoints])
    Z_theo = np.empty([num_types, num_types, num_testpoints])
    for each in range(num_testpoints):
        Y_theo[:, each] = drift(x_init[:, each], num_types, gamma, gamma_prime)
        Z_theo[:, :, each] = diffusion_KM(x_init[:, each], num_types, num_agents, gamma, gamma_prime)

    # Remove wlog last lines
    X = x_init[0:num_types - 1, :]
    Y_theo = Y_theo[0:num_types - 1, :]
    Z_theo = Z_theo[0:num_types - 1, 0:num_types - 1, :]

    # Define observables
    psi = observables.monomials(degree)

    # Evaluate drift and diffusion in X given the data-driven L
    Psi_c = psi(X)

    # Drift coefficients
    # c^0
    b0 = gamma_prime[2, 0]
    # c2^1
    b2 = gamma_prime[1, 0] - gamma_prime[2, 0]
    # c1^1
    b1 = gamma[2, 0] - gamma[0, 2] - gamma_prime[0, 1] - gamma_prime[0, 2] - gamma_prime[2, 0]
    # c1c2
    b4 = gamma[1, 0] - gamma[0, 1] + gamma[0, 2] - gamma[2, 0]
    # c1^2
    b3 = gamma[0, 2] - gamma[2, 0]

    # c^0
    bb0 = gamma_prime[2, 1]
    # c1^1
    bb1 = gamma_prime[0, 1] - gamma_prime[2, 1]
    # c2^1
    bb2 = gamma[2, 1] - gamma[1, 2] - gamma_prime[1, 2] - gamma_prime[1, 2] - gamma_prime[2, 1]
    # c1^2
    bb3 = 0
    # c1c2
    bb4 = gamma[0, 1] - gamma[1, 0] + gamma[1, 2] - gamma[2, 1]
    # c2^2
    bb5 = gamma[1, 2] - gamma[2, 1]

    L_drift_theo = np.array([[b0, b1, b2, b3, b4, 0, 0, 0, 0, 0],
                             [bb0, bb1, bb2, bb3, bb4, bb5, 0, 0, 0, 0]]).T

    # Diffusion coefficients
    # c^0
    a11_0 = 1/num_agents * gamma_prime[2, 0]
    # c1^1
    a11_1 = 1/num_agents * (gamma[0, 2] + gamma[2, 0] + gamma_prime[0, 1] + gamma_prime[0, 2] - gamma_prime[2, 0])
    # c2^1
    a11_2 = 1/num_agents * ( gamma_prime[1, 0] - gamma_prime[2, 0])
    # c1^2
    a11_3 = 1/num_agents * (- gamma[0, 2] - gamma[2, 0])
    # c1c2
    a11_4 = 1/num_agents * (gamma[0, 1] + gamma[1, 0] - gamma[0, 2] - gamma[2, 0] )

    # c^0
    a12_0 = 0
    # c1
    a12_1 = -1/num_agents * gamma_prime[0, 1]
    # c2
    a12_2 = -1/num_agents * gamma_prime[1, 0]
    # c1c2
    a12_4 = -1/num_agents * (gamma[0, 1] + gamma[1, 0])

    # c^0
    a22_0 = 1/num_agents * (gamma_prime[2, 1])
    # c1
    a22_1 = 1/num_agents * (gamma_prime[0, 1] - gamma_prime[2, 1])
    # c2
    a22_2 = 1/num_agents * (gamma[1, 2] + gamma[2, 1] + gamma_prime[1, 0] + gamma_prime[1, 2] - gamma_prime[2, 1])
    # c1^2
    a22_3 = 0
    # c1c2
    a22_4 = 1/num_agents * (gamma[0, 1] + gamma[1, 0] - gamma[1, 2] - gamma[2, 1])
    # c2^2
    a22_5 = 1/num_agents * (-gamma[1, 2] - gamma[2, 1])

    L_diffusion_theo = np.array([[a11_0, a11_1, a11_2, a11_3, a11_4, 0, 0, 0, 0, 0],
                                 [a12_0, a12_1, a12_2, 0, a12_4, 0, 0, 0, 0, 0],
                                 [a22_0, a22_1, a22_2, a22_3, a22_4, a22_5, 0, 0, 0, 0]]).T

    # For all samples
    for j in range(len(num_sample_list)):

        # Load data set
        with np.load(data_path + str(i) + '_' + str(j) + '.npz') as sim_data:
            L = sim_data['L']

        d, V = np.linalg.eig(L)

        L_diffusion_11 = L[:, 3] - 2 * np.array([0, L[0, 1], 0, L[1, 1], L[2, 1], 0, L[3, 1], L[4, 1], 0, 0])

        L_diffusion_12 = (L[:, 4]
                          - np.array([0, L[0, 2], 0, L[1, 2], L[2, 2], 0, 0, L[4, 2], L[5, 2], 0])
                          - np.array([0, 0, L[0, 1], 0, L[1, 1], L[2, 1], 0, L[3, 1], L[4, 1], 0]))

        L_diffusion_22 = L[:, 5] - 2 * np.array([0, 0, L[0, 2], 0, L[1, 2], L[2, 2], 0, 0, L[4, 2], L[5, 2]])

        L_diffusion = np.vstack((L_diffusion_11, L_diffusion_12, L_diffusion_22)).T

        RMSE_drift_coeff[i, j] = RMSE(L_drift_theo, L[:, 1:3])
        RMSE_diffusion_coeff[i, j] = RMSE(L_diffusion_theo, L_diffusion)

        # Drift and diffusion term evaluation using the data-driven generator matrix
        b_c = L[:, 1:num_types].T @ Psi_c
        a_c_11 = L[:, 3].T @ Psi_c - 2 * b_c[0, :] * X[0, :]
        a_c_12 = L[:, 4].T @ Psi_c - b_c[0, :] * X[1, :] - b_c[1, :] * X[0, :]
        a_c_22 = L[:, 5].T @ Psi_c - 2 * b_c[1, :] * X[1, :]

        A = np.empty_like(Z_theo)
        A[0, 0, :] = a_c_11
        A[1, 0, :] = a_c_12
        A[0, 1, :] = a_c_12
        A[1, 1, :] = a_c_22

        RMSE_drift_eval[i, j] = RMSE(Y_theo, b_c)
        RMSE_diffusion_eval[i, j] = RMSE(Z_theo, A)

# %% Single Plots Drift Evaluation

common_min = np.nanmin(RMSE_drift_eval[:, :])
common_max = np.nanmax(RMSE_drift_eval[:, :])

fig = plt.figure()
plt.imshow(RMSE_drift_eval[:, :], norm=colors.LogNorm(vmin=common_min, vmax=common_max))
plt.colorbar(extend='neither')
plt.ylabel('Number of agents $N$')
plt.xlabel('Number of samples $k$')
plt.xticks(np.linspace(0, len(num_sample_list) - 1, len(num_sample_list)), labels=num_sample_list)
plt.yticks(np.linspace(0, len(num_agent_list) - 1, len(num_agent_list)), labels=num_agent_list)
plt.title('Drift evaluation')

# Single Plots Diffusion Evaluation

common_min = np.nanmin(RMSE_diffusion_eval[:, :])
common_max = np.nanmax(RMSE_diffusion_eval[:, :])

fig = plt.figure()

plt.imshow(RMSE_diffusion_eval[:, :], norm=colors.LogNorm(vmin=common_min, vmax=common_max))
plt.colorbar(extend='neither')
plt.ylabel('Number of agents $N$')
plt.xlabel('Number of samples $k$')
plt.xticks(np.linspace(0, len(num_sample_list) - 1, len(num_sample_list)), labels=num_sample_list)
plt.yticks(np.linspace(0, len(num_agent_list) - 1, len(num_agent_list)), labels=num_agent_list)
plt.title('Diffusion evaluation')

# Single Plots Drift Coefficients

common_min = np.nanmin(RMSE_drift_coeff[:, :])
common_max = np.nanmax(RMSE_drift_coeff[:, :])

fig = plt.figure()
plt.imshow(RMSE_drift_coeff[:, :], norm=colors.LogNorm(vmin=common_min, vmax=common_max))
plt.colorbar(extend='neither')
plt.ylabel('Number of agents $N$')
plt.xlabel('Number of samples $k$')
plt.xticks(np.linspace(0, len(num_sample_list) - 1, len(num_sample_list)), labels=num_sample_list)
plt.yticks(np.linspace(0, len(num_agent_list) - 1, len(num_agent_list)), labels=num_agent_list)
plt.title('Drift identification')

# Single Plots Diffusion Coefficients

common_min = np.nanmin(RMSE_diffusion_coeff[:, :])
common_max = np.nanmax(RMSE_diffusion_coeff[:, :])

fig = plt.figure()
plt.imshow(RMSE_diffusion_coeff[:, :], norm=colors.LogNorm(vmin=common_min, vmax=common_max))
plt.colorbar(extend='neither')
plt.ylabel('Number of agents $N$')
plt.xlabel('Number of samples $k$')
plt.xticks(np.linspace(0, len(num_sample_list) - 1, len(num_sample_list)), labels=num_sample_list)
plt.yticks(np.linspace(0, len(num_agent_list) - 1, len(num_agent_list)), labels=num_agent_list)
plt.title('Diffusion identification')
