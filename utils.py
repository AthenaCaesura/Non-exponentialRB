# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:06:58 2021

@author: Athena

General Utilities modele for applying standard RB with a memory which
stores and periodically applies an error.
"""
import numpy as np
from math import sqrt
from functools import lru_cache

single_qubit_paulis = np.array([[[1, 0], [0, 1]],
                                [[0, 1], [1, 0]],
                                [[0, -1j], [1j, 0]],
                                [[1, 0], [0, -1]]], dtype=np.complex128)

H = 1 / sqrt(2) * np.array([[1, 1],
                            [1, -1]])
P = np.array([[1, 0],
              [0, 1j]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])


def get_eigenstate(paulituples):
    """
    Takes list conataining tuples which are (sign, paulinum) where
    sign is either {-1, +1} and paulinum is in {1,2,3} for X, Y, and
    Z respectively. Function returns the pauli eigenstate for each
    of the paulis.

    Parameters
    ----------
    paulituples : list
            List of tuples describing the pauli eigenstate to be made.

    Returns
    -------
    state : numpy array
            State which is the kronecker product of all the paulis
            described by paulituples
    """
    state = np.eye(1)
    for pauli in paulituples:
        state = np.kron(state, 1 / 2 * (np.eye(2) + pauli[0] *
                                        single_qubit_paulis[pauli[1]]))
    return state


def dot(*matrices):
    """
    Compute the matrix product of inputs. Allows for many inputs by recusion.

    Parameters
    ----------
    *matrices : numpy array
            Matricies to be multiplied

    Returns
    -------
    product : numpy array
            Matrix product of the input matricies.

        """
    if (len(matrices) <= 1):
        return matrices[0]
    return np.matmul(matrices[0], dot(*matrices[1:]))


@lru_cache(maxsize=None)
def pauli_on_qubit(pauli_num, qubit_num, num_qubits):
    """
    Creates a matrix which is the pauli given by pauli_num enacted on the
    qubit given by qubit_num.

    Parameters
    ----------
    pauli_num : int in {1,2,3}
            Number designation of the pauli. {1,2,3} -> {X,Y,Z}.
    qubit_num : int in {1,..., num_qubits}
            Number of the qubit that pauli is to act on.
    num_qubits : int
            Number of qubits in the system.

    Returns
    -------
    full_pauli : numpy.array
            The pauli given by pauli_num acting on the qubit given by qubit_num.

    """
    full_pauli = np.eye(1)
    for i in range(num_qubits):
        if i != qubit_num - 1:
            full_pauli = np.kron(full_pauli, np.eye(2))
        else:
            full_pauli = np.kron(full_pauli, single_qubit_paulis[pauli_num])
    return full_pauli


def depolarizing_noise(state, p, n):
    """
    Applied despolarizing noise model to state.

    Parameters
    ----------
    state : numpy array (2^n x 2^n)
            Input state to be depolarized.
    p : double [0,1]
            probability of depolarization
    n : postitive integer
            Number of qubits in system.

    Returns
	-------
	depolarized_state : numpy array (2^n x 2^n)
	            Input state with depolarizing noise attached.
	
	"""
    return 4 / 3 * p * np.eye(2**n) / 2**n + (1 - 4 / 3 * p) * state


def symplectic_to_natural(clif):
    """
    Takes the symplectic form of a clifford gate to the normal
    representaiton of clifford matricies.

    Parameters
    ----------
    clif : np.array
        Clifford Matrix in sympletic representation.

    Returns
    -------
    natural_clif : np.array
        Clifford matrix in natural representation.
    """
    pass
