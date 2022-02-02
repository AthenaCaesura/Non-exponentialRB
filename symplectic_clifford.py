from math import log2
from random import choice
from typing import SupportsAbs
from zlib import Z_BEST_COMPRESSION

import numpy as np
from numpy import *
from numpy import bitwise_xor, dot, inner, row_stack
from numpy.linalg import det, inv

from Sample_Clifford_Element import commutes, random_clifford_generator


def symplectic_product(v, w):
    n = len(v) // 2
    return inner(v[:n], w[n : 2 * n]) != inner(v[:n], w[n : 2 * n])


class SymplecticClifford:
    def __init__(self, symp, global_phase=1):
        n = len(symp) // 2
        if 2 * n != len(symp[0]) - 1 or len(symp) % 2 != 0:
            raise ValueError(f"Size {symp.shape} is not a valid Clifford Operation.")
        self.num_qubits = n
        self.table = np.array(symp, dtype=int)
        self.global_phase = global_phase

    def __mul__(self, other):
        n = self.num_qubits
        new_table = np.dot(self.table[:, :-1], other.table) % 2
        new_table[:, -1] = np.array(
            [i ^ j for i, j in zip(new_table[:, -1], self.table[:, -1])]
        )
        return SymplecticClifford(new_table, self.global_phase * other.global_phase)

    def inv(self):
        augmented_matrix = np.column_stack(
            (self.table[:, :-1], np.eye(2 * self.num_qubits, dtype=int))
        )
        symp_inverse = gf2elim(augmented_matrix)[:, 2 * self.num_qubits :]
        self.table = np.column_stack(
            (
                symp_inverse,
                dot(symp_inverse, self.table[:, -1]),
            )
        )

    def measure_all_qubits(self) -> double:
        prob = 1
        for i in range(self.num_qubits):
            prob *= self.measure_qubit(i)
        return prob

    def measure_qubit(self, a: int) -> double:
        random_measurements = []
        for p in range(self.num_qubits):
            if self.table[self.num_qubits + p, a]:
                random_measurements.append(self.num_qubits + p)

        if random_measurements == []:
            self.table = np.row_stack((self.table, [[0] * (2 * self.num_qubits + 1)]))
            for j in range(self.num_qubits):
                if self.table[a, j]:
                    self.rowsum(2 * self.num_qubits, self.num_qubits + j)
            out = self.table[-1]
            self.table = self.table[:-1]
            return out[-1]
        else:
            p = random_measurements[0]
            for i in random_measurements[1:]:
                self.rowsum(i, p)
            self.table[p - self.num_qubits] = self.table[p]
            self.table[p] = zeros(2 * self.num_qubits + 1)
            num = choice([0, 1])
            self.table[p, -1] = num
            return num

    def rowsum(self, h: int, i: int):
        n = self.num_qubits
        h_row = self.table[h]
        i_row = self.table[i]
        out = zeros(2 * self.num_qubits + 1)
        sum = 2 * h_row[-1] + 2 * i_row[-1]
        for j in range(n):
            sum += self.g(i_row[j], i_row[n + j], h_row[j], h_row[n + j])
            out[j] = h_row[j] ^ i_row[j]
            out[n + j] = h_row[n + j] ^ i_row[n + j]

        if sum % 4 == 2:
            out[-1] = 1

        if sum % 4 == 1 or sum % 4 == 3:
            raise TypeError(
                f"Check if rows {self.table[h]} and {self.table[i]} anticommute "
                f"for rows, {h} and {i} in:\n{self.table}."
            )

        self.table[h] = out

    def g(self, x_1, z_1, x_2, z_2):
        if x_1 and z_1:
            return z_2 - x_2
        if x_1:
            return z_2 & 2 * x_2 - 1
        if z_1:
            return x_2 & 1 - 2 * z_2
        return 0

    def evolve_pauli(self, pauli: np.ndarray):
        """Evolves a given pauli by the clifford represented by self.
            Ignores phases imparted on the pauli by the clifford.

        Args:
            pauli (np.array): Pauli represented in typical tableau format omitting phase.
        """
        return np.dot(self.table[:, :-1], pauli)

    def pauli_mult(self, pauli):
        """Quickly multiplies self.table by a pauli in O(n) time.

        Args:
            pauli (np.array): array encoding a pauli
        """
        new_pauli = [i ^ j for i, j in zip(self.table[:, -1], pauli)]
        new_table = column_stack((self.table[:, :-1], new_pauli))
        return SymplecticClifford(new_table, self.global_phase)

    def assert_commutations(self, msg=""):
        target = np.array(
            [
                [
                    1 if i == j + self.num_qubits else 0
                    for j in range(self.num_qubits * 2)
                ]
                for i in range(self.num_qubits * 2)
            ]
        )
        comm_mat = np.zeros((2 * self.num_qubits, 2 * self.num_qubits))
        for i in range(2 * self.num_qubits):
            for j in range(i):
                comm_mat[i, j] = commutes(self.table[:, :-1], i, j)
        assert det(self.table[:, :-1]) != 0
        assert np.array_equal(comm_mat, target)


# M is a mxn matrix binary matrix
# all elements in M should be uint8
def gf2elim(M):

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
