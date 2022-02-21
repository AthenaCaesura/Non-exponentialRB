from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from NonExponentialRB.Sample_Clifford_Element import random_clifford_generator
from NonExponentialRB.SymplecticClifford import SymplecticClifford, assert_commutations


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
        [[0, 0], [0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ],
)
def test_improperly_sized_table(mat):
    with pytest.raises(ValueError):
        SymplecticClifford(mat)


def test_init_with_global_phase():
    C = SymplecticClifford(np.array([[1, 0, 0], [0, 1, 0]]), global_phase=-1)
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
    C1 = SymplecticClifford(np.array(mat1))
    C2 = SymplecticClifford(np.array(mat2))
    prod = C1 * C2
    assert np.array_equal(prod.table, np.array(target))


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5],
)
def test_commutations_with_random_cliffords(num_qubits):
    for _ in range(1000):
        C = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
        assert_commutations(C.table)


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5],
)
def test_mult_with_random_cliffords(num_qubits):
    for _ in range(100):
        for _ in range(100):
            C_1 = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
            C_2 = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
            assert_commutations((C_1 * C_2).table)


@pytest.mark.parametrize(
    "mat, ans",
    [
        ([[1, 0, 0], [0, 1, 0]], {0: 1, 1: 0}),  # I
        ([[1, 0, 0], [0, 1, 1]], {0: 0, 1: 1}),  # X
        (
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1]],
            {0: 0, 1: 1},
        ),  # X_1 Y_2
    ],
)
def test_measure_qubit(mat, ans):
    num_measurements = 1000
    possible_outcomes = [0, 1]
    measurements = []
    for _ in range(num_measurements):
        C = SymplecticClifford(np.array(mat))
        measurements.append(C.measure())
    measurement_dict = Counter(measurements)
    for outcome in possible_outcomes:
        assert measurement_dict[outcome] / num_measurements == pytest.approx(
            ans[outcome], 0.1
        )


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5],
)
def test_inv(num_qubits):
    for _ in range(1000):
        C = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
        C_original = deepcopy(C)
        C.inv()
        np.testing.assert_array_equal(
            (C_original * C).table,
            np.column_stack((np.identity(2 * num_qubits), np.zeros(2 * num_qubits))),
        )
