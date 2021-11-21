import numpy as np
from Sampler import CliffordSampler
from utils import dot, pauli_on_qubit, depolarizing_noise, comm


def mem_qubit_flip(stored_pauli, n, flip_prob):
    """
    Simulate a faulty memory in register B. Acheives this by multiplying the current
    pauli frame by X's and Z's for each qubit. When an X is applied, we have effectively
    flipped the a qubit in register B which was storing that X gate. Similarly for Z.

    Parameters
    ----------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli stored in register B to be used as the current error applied to
            register A.
    fip_prob : double in [0, 1]
            Probability that a qubit in register B flips due to a memory error.
    n : positive integer
            Number of qubits being benchmarked. (aka the size of register A)

    Returns
    -------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    for qubit_num in range(n):
        if np.random.uniform(0, 1) <= flip_prob:
            X = pauli_on_qubit(1, qubit_num, n)
            stored_pauli = dot(X, stored_pauli)
        if np.random.uniform(0, 1) <= flip_prob:
            Z = pauli_on_qubit(3, qubit_num, n)
            stored_pauli = dot(Z, stored_pauli)
    return stored_pauli


def mem_qubit_reset(stored_pauli, n, reset_prob):
    """
    Simulate a faulty memory in register B. Has a probability of reseting each of
    the qubits in register B to the zero state. Models memory in syndrome qubits.

    Parameters
    ----------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli stored in register B to be used as the current error applied to
            register A.
    reset_prob : double in [0, 1]
            Probability that each pair of qubits in register B corresponding to the
            same qubit in register A resets to |0>.
    n : positive integer
            Number of qubits being benchmarked. (aka the size of register A)

    Returns
    -------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    for qubit_num in range(n):
        # Record how each pauli on this qubit commutes with stored_pauli
        pauli_comms = [
            np.allclose(
                comm(stored_pauli, pauli_on_qubit(i, qubit_num, n)),
                np.zeros(2 ** n),
            )
            for i in range(1, 4)
        ]
        # If X is found on this qubit reset with probability reset_prob
        if (pauli_comms[0] | pauli_comms[1]) & (np.random.uniform(0, 1) <= reset_prob):
            stored_pauli = dot(pauli_on_qubit(1, qubit_num, n), stored_pauli)
        # If Z is found on this qubit reset with probability reset_prob
        if (pauli_comms[2] | pauli_comms[1]) & (np.random.uniform(0, 1) <= reset_prob):
            stored_pauli = dot(pauli_on_qubit(3, qubit_num, n), stored_pauli)
    return stored_pauli


def srb_memory(
    inp_state,
    seq_len,
    init_pauli_error,
    n,
    mem_err_param,
    mem_err_func,
    apply_noise=depolarizing_noise,
    noise_param=0.0,
):
    """
    Run a standard randomized benchmarking (srb) experiment on a qubit register A. While
    doing so, we store a Pauli operator in register B and propagate the pauli through each
    of the gates applied to register A. By applying the pauli stored in register B with
    each gate, we create a noise model which creates a non-exponential decay in the RB
    experiment.

    Parameters
    ----------
    inp_state : (2^n x 2^n) complex numpy array which should be a valid state.
            Initial state for the RB procedure.
    seq_len : positive integer
            Length of the RB sequence excluding inversion gate.
    init_pauli_error : (2^n x 2^n) complex numpy array which should be a Pauli.
            Pauli error stored in register B to be updated with each Clifford
            applied to register A.
    n : positive integer
            Number of qubits to be benchmarked.
    mem_err_param : double in interval [0 1]
            Control parameter for mem_err_func. Usually varied over many calls
            to srb_memory to show differences in memory error.

    Returns
    -------
    p_surv : double in interval [0 1]
            Survival probability of the input state after RB sequence.

    """
    current_state = np.copy(inp_state)
    stored_pauli = np.copy(init_pauli_error)  # Pauli error stored in Register B
    sampler = CliffordSampler(n)
    total_seq = np.eye(2 ** (n))
    for i in range(seq_len):
        C = sampler.sample()
        total_seq = dot(C, total_seq)
        """ Apply noisy random Clifford gate """
        current_state = dot(C, current_state, C.conj().T)
        current_state = apply_noise(current_state, n, noise_param)
        """ Update pauli stored in register B """
        stored_pauli = dot(C, stored_pauli, C.conj().T)
        stored_pauli = mem_err_func(stored_pauli, n, mem_err_param)
        """ Apply pauli stored in register B to register A. """
        current_state = dot(stored_pauli, current_state, stored_pauli.conj().T)
    current_state = dot(total_seq.conj().T, current_state, total_seq)
    p_surv = np.round(np.real(np.trace(np.dot(inp_state, current_state))), 8)
    return p_surv
