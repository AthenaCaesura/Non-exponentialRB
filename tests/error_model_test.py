import pytest
import numpy as np

import sys

from ... import srb

from srb import mem_qubit_flip, mem_qubit_reset
from utils import single_qubit_paulis as sqp
from utils import comm


@pytest.mark.parametrize(
    "input_pauli, correct_output_pauli",
    [(sqp[1], sqp[3]), (sqp[2], sqp[0]), (sqp[3], sqp[1])],
)
def test_certain_mem_qubit_flip(input_pauli, correct_output_pauli):

    output = mem_qubit_flip(input_pauli, 1, 1)
    assert np.allclose(comm(output, correct_output_pauli), np.zeros(2))


@pytest.mark.parametrize(
    "input_pauli",
    [(sqp[1]), (sqp[2]), (sqp[3])],
)
def test_certain_mem_qubit_reset(input_pauli):

    output = mem_qubit_reset(input_pauli, 1, 1)
    """ Ensure [1, 1] element of output is consistent with being the identity matrix """
    assert np.mod(output[1, 1]) == 1 / 2
    output = output / output[1, 1]  # Get rid of global phase
    assert np.allclose(output, sqp[0])
