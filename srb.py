import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, StabilizerState, random_clifford

from utils import comm, depolarizing_noise, dot, pauli_on_qubit


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
            Probability that each memory qubit in register be resets to |0>.
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
    seq_len,
    n,
    mem_err_param,
    mem_err_func,
    init_pauli_error=,
    n_shots = 100,
    noise_param=0.0,
    clifford_noise=lambda x : x,
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
    stored_pauli = np.copy(init_pauli_error)  # Pauli error stored in Register B
    circ = QuantumCircuit(n,n)
    stored_pauli = QuantumCircuit(n,n)
    inv = Clifford.from_label("I" * n) # initalize with identity
    for _ in range(seq_len):
        C = random_clifford(n)
        """ Apply noisy random Clifford gate and track inverse """
        circ.unitary(C, range(n))
        circ = apply_noise(circ, n, noise_param)
        inv &= C
        """ Update pauli stored in register B """
        stored_pauli.unitary(C, range(n))
        stored_pauli = mem_err_func(stored_pauli, n, mem_err_param)
        """ Apply pauli stored in register B to register A"""
        circ.StabilizerState(stored_pauli).to_operator()
    inv = inv.adjoint()
    circ.unitary(inv)
    sim = AerSimulator(method='extended_stabilizer')
    tcirc = transpile(circ, sim)
    result = sim.run(tcirc, nshots)
    return p_surv
