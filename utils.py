# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:06:58 2021

@author: Athena
"""
import numpy as np
from math import sqrt

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
		state = np.kron(state, 1/2 * (np.eye(2) + pauli[0] * 
									 single_qubit_paulis[pauli[1]]))
	return state

def dot(*matrices):
	# Compute the dot product of many matrices, in a recursive fashion, from left to right.
	if (len(matrices) <= 1):
		return matrices[0]
	return np.matmul(matrices[0], dot(*matrices[1:]))