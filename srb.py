import numpy as np
# We need to import the CliffordSampler function.

def Dot(*matrices):
	# Compute the dot product of many matrices, in a recursive fashion, from left to right.
	if (len(matrices) <= 1):
		return matrices[0]
	return np.dot(matrices[0], Dot(matrices[1:]))


def srb_memory(inp_state, seq_len, pauli_frame):
	"""
	Idea:
	1. We want to run a standard randomized benchmarking experiment, on a qubit register A.
	2. While doing so, we want to store a Pauli frame (operator) in register B and track the Pauli frame propagating through each of the Clifford gates applied on register A.

	Inputs:
	a. Input stats: inp_state (2^n x 2^n complex matrix) which should also be a valid state.
	b. Pauli frame: pauli_frame (2^n x 2^n complex matrix), should be a Pauli matrix.
	c. Sequence length: seq_len (a positive integer)

	Output:
	Survival probability (p_surv)

	Method: 
	1. n = size of register A (or the number of qubits in the input state)
	2. current_state = inp_state
	3. current_frame = pauli_frame
	4. For i from 1 to seq_len, do
	5. 		C = CliffordSampler(n)
	6. 		current_state = C . current_state . C^dag
	7. 		current_frame = C . current_frame . C^dag
	8. endfor
	9. p_surv = Tr( inp_state .  current_state )
	10. return p_surv
	"""
	n = int(np.log2(inp_state.dim))
	current_state = inp_state
	current_frame = pauli_frame
	for i in range(seq_len):
		C = CliffordSampler(n)
		current_state = Dot(C, current_state, C.conj().T)
		current_frame = Dot(C, current_frame, C.conj().T)
	p_surv = np.trace(np.dot(inp_state, current_state))
	return p_surv