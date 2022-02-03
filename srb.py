from numpy import argmax, array, bincount, column_stack, identity, newaxis, zeros
from numpy.linalg import inv
from numpy.random import uniform

from Sample_Clifford_Element import random_clifford_generator
from symplectic_clifford import SymplecticClifford


def mem_qubit_flip(reg_b_state, flip_prob):
    """
    Simulate a faulty memory in register B where the encoding for pauli is
    flipped randomly with some probability.

    Parameters
    ----------
    reg_b_state : numpy array (2*n)
            Binary array encoding for a pauli.
    fip_prob : double in [0, 1]
            Probability that a qubit in register B flips due to a memory error.

    Returns
    -------
    err_b_state : numpy array (2*n)
            Binary array encoding for a pauli with error applied.

    """
    err_b_state = []
    for elem in reg_b_state:
        if uniform(0, 1) < flip_prob:
            if uniform(0, 1) < 0.5:
                err_b_state.append(0)
            else:
                err_b_state.append(1)
        else:
            err_b_state.append(elem)
    return array(err_b_state)


def mem_qubit_reset(reg_b_state, reset_prob):
    """
    Simulate a faulty memory in register B where the encoding for pauli is
    erased randomly for each bit with some probability.

    Parameters
    ----------
    reg_b_state : numpy array (2^n x 2^n)
            Binary array encoding for a pauli.
    reset_prob : double in [0, 1]
            Probability that each memory qubit in register be resets to |0>.

    Returns
    -------
    err_b_state : numpy array (2^n x 2^n)
            Binary array encoding for a pauli with error applied.

    """
    err_b_state = []
    for elem in reg_b_state:
        if uniform(0, 1) < reset_prob:
            err_b_state.append(0)
        else:
            err_b_state.append(elem)
    return array(err_b_state)


def srb_memory(seq_len, num_qubits, mem_err_param, mem_err_func, reg_b_copies=1):
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
    reg_b_copies: int
            Copies of register bb to be used to protect memory.

    Returns
    -------
    p_surv : double in interval [0 1]
            Survival probability of the input state after RB sequence.

    """
    reg_a_state = SymplecticClifford(
        column_stack((identity(2 * num_qubits), zeros(2 * num_qubits)))
    )
    reg_b_state = array(
        [[0] * num_qubits + [1] + [0] * (num_qubits - 1)] * reg_b_copies
    )
    tot_seq = SymplecticClifford(
        column_stack((identity(2 * num_qubits), zeros(2 * num_qubits)))
    )
    for _ in range(seq_len):
        C = SymplecticClifford(random_clifford_generator(num_qubits, chp=True))
        """Apply random Clifford gate and track inverse"""
        reg_a_state = C * reg_a_state
        tot_seq = C * tot_seq
        """ Update pauli stored in register B """
        for i in range(reg_b_copies):
            reg_b_state[i] = C.evolve_pauli(reg_b_state[i])
            reg_b_state[i] = mem_err_func(reg_b_state[i], mem_err_param)
        """ Apply pauli stored in register B to register A"""
        majority_vote = [argmax(bincount(pauli)) for pauli in reg_b_state.T]
        reg_b_state = array([majority_vote] * reg_b_copies)
        reg_a_state = reg_a_state.pauli_mult(majority_vote)
    tot_seq.inv()  # invert errorless sequence
    reg_a_state = tot_seq * reg_a_state
    return reg_a_state.measure_all_qubits()  # probability of |00..0> state
