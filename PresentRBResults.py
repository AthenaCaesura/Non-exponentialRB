import multiprocessing
from socket import NI_NAMEREQD

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from srb import mem_qubit_reset, srb_memory


def plot_shots(
    num_qubits,
    maxm=50,
    sample_period=1,
    num_trials=1000,
    mem_err_param=0,
    mem_err_func=mem_qubit_reset,
    filename=None,
):
    """Plots survival probabilities for an RB experiment with an adversarial noise model
    that results in upticks. Stores the results in the Plots folder.

    Args:
        num_qubits (int):
            Number of qubits to be characterized in the RB experiment. Defaults to 1.
        maxm (int, optional):
            Maximum sequence length in the RB experiment. Defaults to 1.
        sample_period (int, optional):
            The rate of sampling of the sequence lengths between 1 and maxm. Defaults
            to 1.
        num_trials (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param (int, optional):
            Controls the severity of the error on the
        mem_err_func (function):
            Error function to be applied to the memory
    """

    """ Gather expectation values for different sequence lengths. """
    tot_evals = np.array([0.0 for _ in range(1, maxm, sample_period)])
    srb_helper = lambda shot_num: _srb_helper(
        shot_num, num_qubits, mem_err_param, mem_err_func, maxm, sample_period
    )
    tot_evals = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(srb_helper)(i) for i in range(num_trials)
    )
    tot_evals = np.mean(np.array(tot_evals).T, axis=1)

    plt.plot(list(range(1, maxm, sample_period)), tot_evals)
    plt.ylabel("Survival Probability")
    plt.xlabel("Gate Sequence Length")
    plt.axis((0, maxm, 0, 1))
    plt.title(
        str(num_qubits)
        + " qubit RB with "
        + str(int(100 * mem_err_param))
        + "% chance of memory error"
    )
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "/"
            + str(int(100 * mem_err_param))
            + "_memory_err_prob_"
            + str(num_qubits)
            + "_qubits"
        )
    plt.savefig("plots/" + filename + ".png")
    plt.clf()


def _srb_helper(shot_num, num_qubits, mem_err_param, mem_err_func, maxm, sample_period):
    """Helper function for parallelization. Indicates the number of shots already taken.

    Args:
        shot_num (int):
            Current shot being taken.
        num_qubits (int):
            Number of qubits to be characterized in the RB experiment. Defaults to 1.
        maxm (int, optional):
            Maximum sequence length in the RB experiment. Defaults to 1.
        sample_period (int, optional):
            The rate of sampling of the sequence lengths between 1 and maxm. Defaults
            to 1.
        num_trials (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param (int, optional):
            Controls the severity of the error on the
        mem_err_func (function):
            Error function to be applied to the memory

    Returns:
        np.array:
            list of binary numbers showing wether the shot failed or succeeded
        for the sequence length indicated by placement in the array.
    """
    print(shot_num)
    return np.array(
        [
            srb_memory(
                seq_len,
                num_qubits,
                mem_err_param,
                mem_err_func,
            )
            for seq_len in range(1, maxm, sample_period)
        ]
    )


plot_shots(
    10,
    num_trials=1000,
    mem_err_param=0.9,
    mem_err_func=mem_qubit_reset,
    filename="tests/test1",
)
