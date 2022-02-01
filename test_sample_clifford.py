from random import random

import numpy as np
import pytest

from Sample_Clifford_Element import commutes, random_clifford_generator
from symplectic_clifford import SymplecticClifford


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5, 20],
)
def test_measure_random_clifford(num_qubits):
    target = np.array(
        [
            [1 if i == j + num_qubits else 0 for j in range(num_qubits * 2)]
            for i in range(num_qubits * 2)
        ]
    )
    for _ in range(1000):
        C = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
        comm_mat = np.zeros((2 * num_qubits, 2 * num_qubits))
        for i in range(2 * num_qubits):
            for j in range(i):
                comm_mat[i, j] = commutes(C.table, i, j)
        assert np.array_equal(comm_mat, target)
