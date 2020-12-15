#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:38:03 2020

@author: jan-hendrikniemann
"""


import numpy as np
import scipy as sp

try:
    from d3s.algorithms import sortEig
except ImportError as err:
    print(err, ('\n\nMake sure you added \'d3s-master to\' '
           'PYTHONPATH. Please add \'D3S - Data-driven dynamical systems '
           'toolbox\' by Stefan Klus to your working environment\n\n--> '
           'https://github.com/sklus/d3s'))
    import sys

    sys.exit()


def gedmd(X, Y, Z, psi, evs=5, operator='K', kramers_moyal=False):
    """
    Generator EDMD for the Koopman operator. The matrices X and Y
    contain the input data. For stochastic systems, Z contains the
    diffusion term evaluated in all data points X. If the system is
    deterministic, set Z = None.

    Modification of 'gedmd' initially implemented in 'D3S - Data-driven
    dynamical systems toolbox' by Stefan Klus: https://github.com/sklus/d3s

    Parameters
    ----------
    X : ndarray
        Input data.
    Y : ndarray
        Input data.
    Z : ndarray
        Input data.
    psi : d3s.observables
        Basis functions.
    evs : int, optional
        Number of eigenvalues to be computed. The default is 5.
    operator : string, optional
        Koopman generator (K) or Perron-Frobenius generator (P).
        The default is 'K'.
    kramers_moyal : bool, optional
        If Z is estimated by the Kramers-Moyal formula, set True. This
        corresponds to Z = sigma * sigma^T. If Z = sigma, set False.
        The default is False.

    Returns
    -------
    A : ndarray
        Matrix approximation of the Koopman or Perron-Frobenius generator.
    d : ndarray
        Eigenvalues of A.
    V : ndarray
        Right eigenvectors of A, which are interpreted as the eigenfunctions
        of the generator.

    """

    PsiX = psi(X)
    dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)
    # stochastic dynamical system
    if not (Z is None):
        # number of basis functions
        n = PsiX.shape[0]
        # second-order derivatives
        ddPsiX = psi.ddiff(X)
        if kramers_moyal:
            S = Z
        else:
            # Compute sigma \cdot sigma^T
            S = np.einsum('ijk,ljk->ilk', Z, Z)
        for i in range(n):
            dPsiY[i, :] += 0.5*np.sum(ddPsiX[i, :, :, :] * S, axis=(0, 1))

    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T
    if operator == 'P':
        C_1 = C_1.T

    A = sp.linalg.pinv(C_0) @ C_1

    if evs > 0:
        d, V = sortEig(A, evs, which='SM')
    else:
        d = None
        V = None

    return (A, d, V)