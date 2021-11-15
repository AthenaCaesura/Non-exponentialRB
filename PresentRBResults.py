from srb import srb_memory
from utils import get_eigenstate, single_qubit_paulis
import matplotlib.pyplot as plt
import numpy as np


def plot_shots(num_qubits=1, maxm=50, sample_period=1, num_trials=1000, mem_fidelity=1):
    """Plots surivial probabilities for an RB experiment with an advarsarial noise model
    that results in upticks. Stores the results in the Plots folder.

    Args:
        num_qubits (int, optional): 
            Number of qubits to be charaterized in the RB experiment. Defaults to 1.
        maxm (int, optional):
            Maximum sequence length in the RB experiment. Defaults to 1.
        sample_period (int, optional):
            The rate of sampling of the sequence lengths between 1 and maxm. Defaults
            to 1.
        num_trials (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_fidelity (int, optional):
            The probability that a qubit in register B retains it's state between
            gate applications on register A. Defaults to 1.
    """
    """ |000..0> is the inital state of register A """
    init_state = get_eigenstate([(1, 3) for _ in range(num_qubits)])
    """ XIII...I is the inital pauli error stored in register B """
    init_pauli_error = np.kron(single_qubit_paulis[1], np.eye(2**(num_qubits - 1)))

    """ Gather expectation values for different sequence lengths. """
    tot_evals = np.array([0. for _ in range(1, maxm, sample_period)])
    for _ in range(num_trials):
        tot_evals += np.array([srb_memory(init_state, num,
                                          init_pauli_error, num_qubits,
                                          mem_fidelity)
                               for num in range(1, maxm, sample_period)])
    tot_evals /= num_trials

    plt.plot(list(range(1, maxm, sample_period)), tot_evals)
    plt.ylabel("Survival Probability")
    plt.xlabel("Gate Sequence Length")
    plt.axis((0, maxm, 0, 1))
    plt.title(str(num_qubits) + " qubit RB with "
              + str(int(100 * mem_fidelity)) + "% memory fidelity")
    plt.savefig("plots/" + str(int(100 * mem_fidelity)) + "_memory_"
                         + str(num_qubits) + "_qubits.png")
    plt.clf()
