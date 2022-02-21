import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from ..Srb_With_Memory import mem_qubit_reset, srb_with_memory


def plot_expectation_values(
    num_qubits,
    max_seq_length=50,
    sample_period=1,
    mem_err_param=0,
    mem_err_func=mem_qubit_reset,
    reg_b_copies=1,
    correction_on_reg_b=True,
    num_samples=1000,
    filename=None,
    show_plot=False,
):
    """Plots survival probabilities for an RB experiment with an adversarial noise model
    that results in upticks. Stores the results in the Plots folder.
    Utilizes Multithreading.

    Args:
        num_qubits (int):
            Number of qubits to be characterized in the RB experiment. Defaults to 1.
        max_seq_length (int, optional):
            Maximum sequence length in the RB experiment. Defaults to 1.
        sample_period (int, optional):
            The rate of sampling of the sequence lengths between 1 and max_seq_length.
            Defaults to 1.
        num_samples (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param (int, optional):
            Controls the severity of the error on the
        mem_err_func (function):
            Error function to be applied to the memory
        reg_b_copies (int):
            Number of copies of register b used as a memory.
        filename (directory):
            Name of file to save results in. Always saves in plots.
        show_plot (bool):
            If true, then run matplotlib.pyplot.show as soon as plot is created.

    """

    """ Gather expectation values for different sequence each length. """
    evals = np.array([0.0 for _ in range(1, max_seq_length, sample_period)])
    helper = partial(
        _srb_with_memory_helper,
        num_qubits,
        max_seq_length,
        sample_period,
        mem_err_param,
        mem_err_func,
        reg_b_copies,
        correction_on_reg_b,
    )
    evals = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(helper)(sample_num) for sample_num in range(num_samples)
    )
    """ Average the collected shots into expectation values for each
    sequence length"""
    evals = np.mean(np.array(evals).T, axis=1)

    """ Plots a line graph with each of the expectation values """
    plt.plot(list(range(1, max_seq_length, sample_period)), evals)
    plt.ylabel("Survival Probability")
    plt.xlabel("Gate Sequence Length")
    plt.axis((0, max_seq_length, 0, 1))
    plt.title(
        str(num_qubits)
        + " qubit RB with "
        + str(int(100 * mem_err_param))
        + "% chance of memory error"
    )
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "_"
            + str(int(100 * mem_err_param))
            + "_memory_err_prob_"
            + str(num_qubits)
            + "_qubits"
        )
    plt.savefig("plots/" + filename + ".png")
    if show_plot:
        plt.show()
    plt.clf()

    return list(zip(range(1, max_seq_length, sample_period), evals))


def _srb_with_memory_helper(
    num_qubits,
    max_seq_length,
    sample_period,
    mem_err_param,
    mem_err_func,
    reg_b_copies,
    correction_on_reg_b,
    sample_num,
):
    """Helper function for parallelization. Each loop prints a sample for each intger
    less than max_seq_length.


    Args:
        num_qubits (int):
            Number of qubits to be characterized in the RB experiment. Defaults to 1.
        max_seq_length (int, optional):
            Maximum sequence length in the RB experiment. Defaults to 1.
        sample_period (int, optional):
            The rate of sampling of the sequence lengths between 1 and max_seq_length.
            Defaults to 1.
        num_samples (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param (int, optional):
            Controls the severity of the error on the
        mem_err_func (function):
            Error function to be applied to the memory
        sample_num (int):
            Current shot being taken to calculate expectation value.

    Returns:
        np.array:
            List of binary numbers indicating wether the shot failed or succeeded. For
            each list index * length sample_period = length of sequence sampled.
    """
    print(sample_num)
    return np.array(
        [
            srb_with_memory(
                seq_len,
                num_qubits,
                mem_err_param,
                mem_err_func,
                reg_b_copies=reg_b_copies,
                correction_on_reg_b=correction_on_reg_b,
            )
            for seq_len in range(1, max_seq_length, sample_period)
        ]
    )
