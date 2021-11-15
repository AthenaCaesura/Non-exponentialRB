import numpy as np
from Sampler import CliffordSampler
from utils import dot, pauli_on_qubit, depolarizing_noise


def faulty_qubit_mem(stored_pauli, mem_fidelity, n):
    """
    Simulate a faulty memory in regiester B. Acheives this by multiplying the current
    pauli frame by X's and Z's for each qubit. When an X is applied, we have effectively
    flipped the a qubit in register B which was storing that X gate. Similarly for Z.

    Parameters
    ----------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli stored in register B to be used as the current error applied to
            register A.
    mem_fidelity : double in [0, 1]
            Probability that a qubit in register B doesn't flips due to a memory
            error.
    n : positive integer
            Number of qubits being benchmarked. (aka the size of register A)

    Returns
    -------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    for qubit_num in range(n):
        if np.random.uniform(0, 1) >= mem_fidelity:
            X = pauli_on_qubit(1, qubit_num + 1, n)
            stored_pauli = dot(X, stored_pauli)
        if np.random.uniform(0, 1) >= mem_fidelity:
            Z = pauli_on_qubit(3, qubit_num + 1, n)
            stored_pauli = dot(Z, stored_pauli)
    return stored_pauli


def srb_memory(inp_state, seq_len, init_pauli_error, n, mem_fidelity,
               mem_err_func=faulty_qubit_mem,
               apply_noise=depolarizing_noise,
               noise_param=.0):
    """
    Run a standard randomized benchmarking experiment on a qubit register A. While doing
    so, we store a Pauli operator in register B and propogate the pauli through each of
    the gates applied to register A. By applying the pauli stored in register B with
    each gate, we create a noise model which creates a non-exponential decay in the RB
    experiment.

    Parameters
    ----------
    inp_state : (2^n x 2^n) complex numpy array which should be a valid state.
            Initial state for the RB procedure.
    seq_len : postitive integer
            Length of the RB sequence exluding inversion gate.
    init_pauli_error : (2^n x 2^n) complex numpy array which should be a Pauli.
            Pauli error stored in register B to be updated with each Clifford
            applied to register A.
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
    stored_pauli = np.copy(init_pauli_error) # Pauli error stored in Register B
    sampler = CliffordSampler(n)
    total_seq = np.eye(2**(n))
    for i in range(seq_len):
        C = sampler.sample()
        total_seq = dot(C, total_seq)
        """ Apply noisy random Clifford gate """
        current_state = dot(C, current_state, C.conj().T)
        current_state = apply_noise(current_state, noise_param, n)
        """ Update pauli stored in register B """
        stored_pauli = dot(C, stored_pauli, C.conj().T)
        stored_pauli = mem_err_func(stored_pauli, mem_fidelity, n)
        """ Apply pauli stored in register B to register A. """
        current_state = dot(stored_pauli, current_state, stored_pauli.conj().T)
    current_state = dot(total_seq.conj().T, current_state, total_seq)
    p_surv = np.round(np.real(np.trace(np.dot(inp_state, current_state))), 8)
    return p_surv
