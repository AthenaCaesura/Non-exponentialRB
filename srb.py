import numpy as np
from qiskit.quantum_info import Clifford, Pauli, StabilizerState, random_clifford

MULT_SINGLE_QUBIT_PAULIS_NO_PHASE = {
    "I": {
        "I": "I",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
    },
    "X": {
        "I": "X",
        "X": "I",
        "Y": "Z",
        "Z": "Y",
    },
    "Y": {
        "I": "Y",
        "X": "Z",
        "Y": "I",
        "Z": "X",
    },
    "Z": {
        "I": "Z",
        "X": "Y",
        "Y": "X",
        "Z": "I",
    },
}


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

    Returns
    -------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    stored_pauli_with_error = ""
    for pauli_str in stored_pauli.to_label():
        if np.random.uniform(0, 1) <= flip_prob:
            MULT_SINGLE_QUBIT_PAULIS_NO_PHASE[pauli_str]["X"] = pauli_str
        if np.random.uniform(0, 1) <= flip_prob:
            MULT_SINGLE_QUBIT_PAULIS_NO_PHASE[pauli_str]["Z"] = pauli_str
        stored_pauli_with_error += pauli_str
    return Pauli(stored_pauli_with_error)


def mem_qubit_reset(stored_pauli, reset_prob):
    """
    Simulate a faulty memory in register B. Has a probability of reseting each of
    the qubits in register B to the zero state. Models memory in syndrome qubits.

    Parameters
    ----------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli stored in register B to be used as the current error applied to
            register A.
    reset_prob : double in [0, 1]
            Probability that each memory qubit in register be resets to |0>.

    Returns
    -------
    stored_pauli : numpy array (2^n x 2^n)
            Pauli matrix used as the current error with faulty memory applied.

    """
    stored_pauli_with_error = ""
    for pauli_str in stored_pauli.to_label():
        # If X is found on this qubit, reset with probability reset_prob
        if np.random.uniform(0, 1) <= reset_prob:
            if pauli_str == "X":
                pauli_str = "I"
            if pauli_str == "Y":
                pauli_str = "Z"
        # If Z is found on this qubit, reset with probability reset_prob
        if np.random.uniform(0, 1) <= reset_prob:
            if pauli_str == "Z":
                pauli_str = "I"
            if pauli_str == "Y":
                pauli_str = "X"
        stored_pauli_with_error += pauli_str
    return Pauli(stored_pauli_with_error)


def srb_memory(seq_len, n, mem_err_param, mem_err_func):
    """
    Run a standard randomized benchmarking (srb) experiment on a qubit register A. While
    doing so, we store a Pauli operator in register B and propagate the pauli through each
    of the gates applied to register A. By applying the pauli stored in register B with
    each gate, we create a noise model which creates a non-exponential decay in the RB
    experiment.

    Parameters
    ----------
    seq_len : positive integer
            Length of the RB sequence excluding inversion gate.
    n : positive integer
            Number of qubits to be benchmarked.
    mem_err_param : double in interval [0 1]
            Control parameter for mem_err_func. Usually varied over many calls
            to srb_memory to show differences in memory error.
    mem_err_func : function (Pauli, mem_err_param) -> Pauli
            Error applied to the pauli stored in register b due to faulty memory.

    Returns
    -------
    p_surv : double in interval [0 1]
            Survival probability of the input state after RB sequence.

    """
    reg_a_state = StabilizerState(Pauli("I" * n))  # initialize in |00..0> state
    pauli_stored_in_reg_b = Pauli("X" + "I" * (n - 1))
    total_seq = Clifford.from_label("I" * n)
    for _ in range(seq_len):
        C = random_clifford(n)
        """ Apply noisy random Clifford gate and track inverse """
        reg_a_state = reg_a_state.evolve(C)
        total_seq &= C
        """ Update pauli stored in register B """
        pauli_stored_in_reg_b = pauli_stored_in_reg_b.evolve(C)
        pauli_stored_in_reg_b = mem_err_func(pauli_stored_in_reg_b, mem_err_param)
        """ Apply pauli stored in register B to register A"""
        reg_a_state = reg_a_state.evolve(
            Clifford.from_label(pauli_stored_in_reg_b.to_label().replace("-", ""))
        )
    inv = total_seq.adjoint()
    reg_a_state = reg_a_state.evolve(inv)
    return reg_a_state.probabilities()[0]  # probability of |00..0> state


from time import time

start = time()
srb_memory(1, 1, 0, mem_qubit_reset)
print(start - time())

start = time()
srb_memory(1000, 1, 0, mem_qubit_reset)
print(time() - start)
