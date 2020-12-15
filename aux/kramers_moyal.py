#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:43:37 2020

@author: Jan-Hendrik Niemann
"""

import numpy as np


def kramersmoyal(t, X):
    """
    Kramers-Moyal estimation of drift and diffusion term

    Parameters
    ----------
    t : float
        Lag time.
    X : ndarray
        Of type [dimension, num_timesteps, num_samples, num_testpoints].

    Returns
    -------
    b_X : float
        Estimation of the drift evaluated at X.
    a_X : float
        Estimation of the diffusion^2 evaluated at X.

    """

    dimension = X.shape[0]
    num_samples = X.shape[2]
    num_testpoints = X.shape[3]

    # difference has shape [dimension, num_samples, num_testpoints]
    difference = X[:, -1, :, :] - X[:, 0, :, :]

    # b_X has shape [dimension, num_testpoints]
    b_X = 1/t * np.mean(difference, axis=1)

    product = np.zeros([dimension, dimension, num_testpoints, num_samples])
    for i in range(num_testpoints):
        for j in range(num_samples):
            product[:, :, i, j] = np.outer(difference[:, j, i],
                                           difference[:, j, i])

    # a_X has shape [dimension, dimension, num_testpoints]
    a_X = 1/t * np.mean(product, axis=3)

    return b_X, a_X

# %% Test case from "Data-driven approximation of the Koopman generator:
# Model reduction, system identication, and control" by S. Klus, F. Nüske,
# S. Peitz, J.-H. Niemann, C. Clementi and C. Schütte


if __name__ == '__main__':

    # 2d Example
    def drift(x):
        return np.array([-4*x[0]**3 + 4*x[0], -2*x[1]])

    def diff(x):
        return np.array([[0.7, x[0]], [0, 0.5]])

    x_0 = np.array([np.random.uniform(-2, 2), np.random.uniform(-1, 1)])

    num_samples = int(1e5)
    T_max = 1e-4
    t_step = 1e-5

    num_timesteps = int(np.round(T_max / t_step)) + 1

    X = np.empty([2, 2, num_samples, 1])

    for i in range(num_samples):
        X[:, 0, i, 0] = x_0

    for j in range(num_samples):
        x_old = x_0
        x_new = np.empty_like(x_old)
        for i in range(num_timesteps):
            dW = np.random.normal(0, np.sqrt(t_step), 2)
            x_new = x_old + drift(x_old)*t_step + diff(x_old) @ dW
            x_old = x_new
        X[:, 1, j, 0] = x_new

    b, a = kramersmoyal(T_max, X)

    b_theory = drift(x_0)
    a_theory = diff(x_0) @ diff(x_0).T

    print('Error drift term b:\n', np.abs(b[:, 0] - b_theory))
    print('Error diffusion term a:\n', np.abs(a[:, :, 0] - a_theory))

# %% 1d Example

    num_samples = int(1e5)
    T_max = 1e-4
    t_step = 1e-5

    def drift2(x, alpha=1):
        return -alpha*x

    def diff2(x, beta=4):
        return np.sqrt(2/beta)

    num_timesteps = int(np.round(T_max / t_step)) + 1

    # X -> [dimension, num_timesteps, num_samples, num_testpoints]
    X = np.empty([1, num_timesteps, num_samples, 1])
    x_0 = np.random.uniform(-2, 2)

    for j in range(num_samples):
        X[0, 0, j, 0] = x_0
        for i in range(num_timesteps-1):
            dW = np.random.normal(0, np.sqrt(t_step))
            X[0, i+1, j, 0] = X[0, i, j, 0] + drift2(X[0, i, j, 0]) * t_step
            + diff2(X[0, i, j, 0]) * dW

    b, a = kramersmoyal(T_max, X)

    b_theory = drift2(x_0)
    a_theory = diff2(x_0) ** 2

    print('Error drift term b:\n', np.abs(b - b_theory).item())
    print('Error diffusion term a:\n', np.abs(a - a_theory).item())
