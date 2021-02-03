#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:03:35 2020

Auxiliary functions.

@author: Jan-Hendrik Niemann
"""


import numpy as np


def RMSE(A, A_tilde):
    """
    Root mean square error. Gives the standard deviation of the residuals
    (prediction errors).

    Parameters
    ----------
    A : ndarray
        Forecast.
    A_tilde : ndarray
        Observation.

    Returns
    -------
    float
        Root mean square error.

    """
    return np.sqrt((abs(A_tilde - A)**2).mean())


def random_init(N, k, s=1):
    """
    Multinomial distributed vector of k integer between [0, N] summing up to N

    Parameters
    ----------
    N : int
        Upper bound of the drawing set [0, N].
    k : int
        Sample size.
    s : float, optional
        Controls smoothness of distributions. Larger values of s cause
        approach Poisson with mean N/k and smaller values lead to greater
        variance. The default is 1 and corresponds to a uniform distribution.

    Returns
    -------
    ndarray
        Vector of length k with multinomial disrtibuted integers between [0, N]

    """
    return np.random.multinomial(N, s * np.random.dirichlet(np.ones(k)))


def random_init_2(k, rng=None):
    """
    Uniformly distributed float vector of length k between summing up to 1

    Parameters
    ----------
    k : int
        Sample size.
    rng : Numpy random number generator
        The default is None.

    Returns
    -------
    ndarray
        Vector of k uniformly distributed floats on the standard simplex.

    """
    # New random number generator
    if rng is None:
        rng = np.random.default_rng()

    return rng.dirichlet(np.ones(k))
