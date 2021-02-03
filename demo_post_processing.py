#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:55:10 2020

Computes all generator approximation matrices L by gEDMD for the three types
case.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import os

import d3s.observables as observables
import auxiliary.algorithms as algorithms
from auxiliary.kramers_moyal import kramersmoyal


# Maximum degree of monomials
degree = 3

data_path = 'data/raw/'
save_path = 'data/processed/matrix_'

# Define observables
psi = observables.monomials(degree)

for filename in os.listdir(data_path):

    if filename[-3::] != 'npz':
        continue

    with np.load(data_path + filename) as data:
        print('Available in this data set:', data.files)
        trajectory = data['trajectory']
        x_init = data['x_init']
        num_agents = data['num_agents']
        num_testpoints = data['num_trainingpoints']
        num_samples = data['num_samples']
        T_max = data['T_max']
        gamma = data['gamma']
        gamma_prime = data['gamma_prime']

    num_types = x_init.shape[0]

    print('\nNumber of agents:', int(num_agents))
    print('Number of Monte Carlo samples for Kramers-Moyal:', num_samples)
    print('Number of testpoints for gEDMD:', num_testpoints)
    print(25 * '= ')

    # Rescale data to unit interval
    trajectory /= num_agents
    x_init /= num_agents

    # Remove wlog last lines
    trajectory = trajectory[0:num_types - 1, :, :, :]
    X = x_init[0:num_types - 1, :]
    b_X, a_X = kramersmoyal(T_max, trajectory)

    # Apply generator gEDMD
    # Number of eigenvalues/eigenfunctions to be computed
    evs = 0
    # Get Koopman generator approximation
    L, __, __ = algorithms.gedmd(X, b_X, a_X, psi, evs=evs, operator='K', kramers_moyal=True)
    P, __, __ = algorithms.gedmd(X, b_X, a_X, psi, evs=evs, operator='P', kramers_moyal=True)

    # Save Koopman generator and Perron-Forbenius generator
    num_str = filename[4::]
    np.savez_compressed(save_path + num_str,
                        L=L,
                        P=P,
                        num_agents=num_agents,
                        num_testpoints=num_testpoints,
                        num_samples=num_samples,
                        num_types=num_types,
                        T_max=T_max,
                        degree=degree,
                        gamma=gamma,
                        gamma_prime=gamma_prime)
