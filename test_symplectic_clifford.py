from collections import Counter
from copy import deepcopy
from re import S

import numpy as np
import pytest

from Sample_Clifford_Element import random_clifford_generator
from symplectic_clifford import SymplecticClifford


@pytest.mark.parametrize(
    "mat",
    [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0]],
        [[1, 1, 0], [0, 1, 0]],
    ],
)
def test_init(mat):
    C = SymplecticClifford(mat)
    assert np.array_equal(C.table, mat)
    assert C.num_qubits == len(mat) // 2
    assert C.global_phase == 1


@pytest.mark.parametrize(
    "mat",
    [
        [[0, 0, [0, 0]], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ],
)
def test_improperly_sized_table(mat):
    with pytest.raises(ValueError):
        SymplecticClifford(mat)


def test_init_global_phase():
    C = SymplecticClifford([[1, 0, 0], [0, 1, 0]], global_phase=-1)
    assert C.global_phase == -1


@pytest.mark.parametrize(
    "mat1, mat2, target",
    [
        ([[1, 0, 0], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]]),
        ([[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]),
        (
            [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]],
            [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]],
        ),
    ],
)
def test_mult(mat1, mat2, target):
    C1 = SymplecticClifford(mat1)
    C2 = SymplecticClifford(mat2)
    prod = C1 * C2
    assert np.array_equal(prod.table, target)


@pytest.mark.parametrize(
    "mat, summed",
    [
        (
            [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]],
        ),
        (
            [[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]],
        ),
    ],
)
def test_rowsum(mat, summed):
    C = SymplecticClifford(mat)
    C.rowsum(0, 1)
    breakpoint()
    assert np.array_equal(C.table, summed)


@pytest.mark.parametrize(
    "mat, ans",
    [
        ([[1, 0, 0], [0, 1, 0]], {0: 1, 1: 0}),  # I
        ([[1, 0, 0], [0, 1, 1]], {0: 0, 1: 1}),  # X
        ([[0, 1, 0], [1, 0, 0]], {0: 0.5, 1: 0.5}),  # H
    ],
)
def test_measure_qubit(mat, ans):
    num_measurements = 1000
    possible_outcomes = [0, 1]
    measurements = []
    for _ in range(num_measurements):
        C = SymplecticClifford(mat)
        measurements.append(C.measure_qubit(0))
    measurement_dict = Counter(measurements)
    for outcome in possible_outcomes:
        assert measurement_dict[outcome] / num_measurements == pytest.approx(
            ans[outcome], 0.1
        )


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5, 20],
)
def test_inv(num_qubits):
    for _ in range(100):
        C = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
        C_original = deepcopy(C)
        C.inv()
        assert np.array_equal(
            (C_original * C).table,
            np.column_stack((np.identity(2 * num_qubits), np.zeros(2 * num_qubits))),
        )
