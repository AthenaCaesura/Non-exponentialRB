"""
All credit for this function goes to  Samuele Cornell.
"""

import numpy as np


def gf2elim(M: np.ndarray):
    """
    Args:
        M (np.array): binary matrix to perform  gaussian elimination on.
    All elements in M should be uint8.

    Returns:
        (np.ndarray) : Matrix with gaussian elimination over f2 performed on it.
    """

    m, n = M.shape
    i = 0
    j = 0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) + i

        # swap rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j])  # make a copy otherwise M will be directly affected

        col[i] = 0  # avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j += 1

    return M
