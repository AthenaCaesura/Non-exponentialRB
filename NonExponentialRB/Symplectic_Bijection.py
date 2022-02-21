"""
Python code from "How to efficiently select and arbitrary
clifford group element. " Credit for this file goes to
Robert Keonig and John A. Smolin from the above paper.

The bijection is primarily used in this repository to sample
a random element of the symplectic group which is used in
sampling a random element of the clifford group.
"""

import numpy as np


def directsum(m1, m2):
    n1 = len(m1[0])
    n2 = len(m2[0])
    output = np.zeros((n1 + n2, n1 + n2), dtype=np.int8)
    for i in range(0, n1):
        for j in range(0, n1):
            output[i, j] = m1[i, j]
    for i in range(0, n2):
        for j in range(0, n2):
            output[i + n1, j + n1] = m2[i, j]
    return output


def inner(v, w):
    t = 0
    for i in range(0, np.size(v) >> 1):
        t += v[2 * i] * w[2 * i + 1]
        t += w[2 * i] * v[2 * i + 1]
    return t % 2


def transvection(k, v):
    return (v + inner(k, v) * k) % 2


def int_to_bits(i, n):
    """Converts an integer i to an array of n bits representing
    a binary equivalent to i.

    Args:
        i (int): To convert to array of bits
        n (int): Number of bits from to convert i into

    Returns:
        output (List[bit]): List of bits equivalent to i.
    """
    output = np.zeros(n, dtype=np.int8)
    for j in range(0, n):
        output[j] = i & 1
        i >>= 1
    return output


def find_transvection(x, y):
    """Find h1 and h2 such that y = Z_h1 Z_h1 x
    Lemma 2 in the text.
    """
    output = np.zeros((2, np.size(x)), dtype=np.int8)
    if np.array_equal(x, y):
        return output
    if inner(x, y) == 1:
        output[0] = (x + y) % 2
        return output
    # find a pair where they are both not 00
    z = np.zeros(np.size(x))
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) != 0):
            z[ii] = (x[ii] + y[ii]) % 2
            z[ii + 1] = (x[ii + 1] + y[ii + 1]) % 2
            if (z[ii] + z[ii + 1]) == 0:  # they were the same so they added t o 00
                z[ii + 1] = 1
            if x[ii] != x[ii + 1]:
                z[ii] = 1
            output[0] = (x + z) % 2
            output[1] = (y + z) % 2

    # didn't find a pair so look for two places where x has 00 and
    # y doesn't and vice versa
    #
    # first y == 00 and x doesn't
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) == 0):  # found the pair
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break
    # finally x == 00 and y doesn't
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) == 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break
    output[0] = (x + z) % 2
    output[1] = (y + z) % 2
    return output


def symplectic_bijection(i, n):
    """Bijection between the symplectic matrices of size n and integers.
    Returns the integer labeled i in the bijection.

    Args:
        i (int): number of matrix in enumeration of symplectic group.
        n (int): number of qubits in representation
    """
    nn = 2 * n
    # step 1
    s = (1 << nn) - 1
    k = (i % s) + 1
    i //= s

    # step 2
    f1 = int_to_bits(k, nn)

    # step 3
    e1 = np.zeros(nn, dtype=np.int8)  # define first basis vectors
    e1[0] = 1
    T = find_transvection(e1, f1)  # use lemma 2 to compute T

    # step 4
    bits = int_to_bits(i % (1 << (nn - 1)), nn - 1)

    # step 5
    eprime = np.copy(e1)
    for j in range(2, nn):
        eprime[j] = bits[j - 1]
    h0 = transvection(T[0], eprime)
    h0 = transvection(T[1], h0)

    # step 6
    if bits[0] == 1:
        f1 *= 0

    # step 7
    id2 = np.identity(2, dtype=np.int8)
    if n != 1:
        g = directsum(id2, symplectic_bijection(i >> (nn - 1), n - 1))
    else:
        g = id2
    for j in range(0, nn):
        g[j] = transvection(T[0], g[j])
        g[j] = transvection(T[1], g[j])
        g[j] = transvection(h0, g[j])
        g[j] = transvection(f1, g[j])
    return g
