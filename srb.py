import numpy as np
from Sampler import CliffordSampler
from utils import dot

def srb_memory(inp_state, seq_len, pauli_frame, n):
	"""
	Idea:
	1. We want to run a standard randomized benchmarking experiment, on a
       qubit register A.
	2. While doing so, we want to store a Pauli frame (operator) in register B
       and track the Pauli frame propagating through each of the Clifford
       gates applied on register A.

	Inputs:
	a. Input stats: inp_state (2^n x 2^n complex matrix) which should also be a valid state.
	b. Pauli frame: pauli_frame (2^n x 2^n complex matrix), should be a Pauli matrix.
	c. Sequence length: seq_len (a positive integer)
	d. Size of Register A

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
	current_state = np.copy(inp_state)
	current_frame = np.copy(pauli_frame)
	sampler = CliffordSampler(n)
	total_seq = np.eye(2**(n))
	for i in range(seq_len):
		C = sampler.sample()
		total_seq = dot(C, total_seq)
		current_state = dot(C, current_state, C.conj().T)
		current_state = apply_noise(current_state, .0, n)
		current_frame = dot(C, current_frame, C.conj().T)
		if np.random.uniform(0,1) <= .95:
			current_state = dot(current_frame, current_state,
							    current_frame.conj().T) # apply frame
	current_state = dot(total_seq.conj().T, current_state, total_seq)
	p_surv = np.round(np.real(np.trace(np.dot(inp_state, current_state))),8)
	return p_surv

def apply_noise(state, p, n):
	return 4/3 * p * np.eye(2**n)/2**n + (1- 4/3 * p) * state