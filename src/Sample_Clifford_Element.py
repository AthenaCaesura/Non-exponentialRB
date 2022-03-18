from functools import lru_cache
from random import choices, randrange

import numpy as np

from Symplectic_Bijection import symplectic_bijection


def random_clifford_generator(n_qubits, chp=False):
    """Returns matrix representing a randomly selected clifford operation
    on num_qubit. The matrix is of the form (symplectic matrx| pauli vector).
    The CHP option will return the sympletic matrix in the form given in
    "Improved Stabilizer circuit simulation" by Aaronson and Gottesman.

    Args:
        n_qubits (int): Number of qubits random clifford acts on.
        chp (bool, optional): If true, return the sympletic matrix in the
        form given in "Improved Stabilizer circuit simulation" by
        Aaronson and Gottesman. If false, return the sympletic matrix in the
        form given in "How to efficiently select and arbitrary clifford group
        element" by Keonig and Smolin.Defaults to False.

    Returns:
        np.ndarray: (2*num_qubits x 2*num_qubits + 1) matrix representing a
        random clifford operator.
    """
    symp_index = int(randrange(order_of_symplectic_group(n_qubits)))
    symp = np.array(symplectic_bijection(symp_index, n_qubits), dtype=np.int8)
    if chp:
        symp = chp_format(symp)
    pauli_string = choices([0, 1], k=2 * n_qubits)
    return np.column_stack((symp, pauli_string))


def chp_format(symp):
    """Transform Symplectic representation given in "How to efficiently
    select and arbitrary clifford group element" by Keonig and Smolin
    to the representation used in "Improved Stabilizer circuit simulation"
    by Aaronson and Gottesman.

    Args:
        symp (np.ndarray): Symplectic matrix in representation used in
        "How to efficiently select and arbitrary clifford group element."

    Returns:
        symp (np.ndarray): Symplectic matrix in representation used in
        "Improved Stabilizer circuit simulation"
    """
    n = len(symp) // 2
    perm = list(range(2 * n))
    perm = perm[0::2] + perm[1::2]
    return symp[perm][:, perm]


@lru_cache(maxsize=1)
def order_of_symplectic_group(n_qubits):
    """Returns order of the n_qubit syplectic group.

    Args:
        n_qubits (int): Number of qubits in symplectic representation.

    Returns:
        _type_: _description_
    """
    # num = 2 ** (n_qubits ** 2)
    # for i in range(1, n_qubits):
    #     num *= 4 ** i - 1
    order = np.power(2, n_qubits * n_qubits, dtype = np.float64) * np.prod(np.power(4, np.arange(1, n_qubits, dtype = np.float64), dtype = np.float64))
    # print("order_of_symplectic_group({}) = {}".format(n_qubits, order))
    return order
