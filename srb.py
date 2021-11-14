import numpy as np
from Sampler import CliffordSampler
from utils import dot, pauli_on_qubit, depolarizing_noise


def faulty_qubit_mem(current_frame, mem_fidelity, n):
    """
    Simulate a faulty memory in regiester B. Acheives this by multiplying
    the current pauli frame by X's and Z's for each qubit. When an X is
    applied, we have effectively flipped the a qubit in register B which
    was storing that X gate. Similarly for Z.

    Parameters
    ----------
    current_frame : numpy array (2^n x 2^n)
            Pauli matrix to be used as the current error applied to register A.
    mem_fidelity : double in [0, 1]
            Probability that a qubit in register B doesn't flips due to a memory
            error.
    n : positive integer
            Number of qubits being benchmarked. (aka the size of register A)

    Returns
    -------
    current_frame : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    for qubit_num in range(n):
        if np.random.uniform(0, 1) >= mem_fidelity:
            X = pauli_on_qubit(1, qubit_num + 1, n)
            current_frame = dot(X, current_frame)
        if np.random.uniform(0, 1) >= mem_fidelity:
            Z = pauli_on_qubit(3, qubit_num + 1, n)
            current_frame = dot(Z, current_frame)
    return current_frame


def srb_memory(inp_state, seq_len, pauli_frame, n, mem_fidelity,
               mem_err_func=faulty_qubit_mem,
               apply_noise=depolarizing_noise):
    """
    Run a standard randomized benchmarking experiment on a
    qubit register A. While doing so, we want to store a Pauli frame
    (operator) in register B and track the Pauli frame propagating
    through each of the Clifford gates applied on register A.

    Parameters
    ----------
    inp_state : (2^n x 2^n) complex numpy array which should be a valid state.
            Initial state for the RB procedure.
    seq_len : postitive integer
            Length of the RB sequence exluding inversion gate.
    pauli_frame : (2^n x 2^n) complex numpy array which should be a Pauli.
            Pauli frame to be updated with each Clifford application.
    n : positive integer
            Number of qubits to be benchmarked.
    mem_prob : double in interval [0 1]
            Probability that the each qubit in register b which stores the pauli
            frame flips.

    Returns
    -------
    p_surv : double in interval [0 1]
            Survival probability of the input state after RB sequence.

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
        current_frame = mem_err_func(current_frame, mem_fidelity, n)
        current_state = dot(current_frame, current_state,  # apply frame
                            current_frame.conj().T)
    current_state = dot(total_seq.conj().T, current_state, total_seq)
    p_surv = np.round(np.real(np.trace(np.dot(inp_state, current_state))), 8)
    return p_surv
