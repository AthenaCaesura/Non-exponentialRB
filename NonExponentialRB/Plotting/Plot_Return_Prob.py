import matplotlib.pyplot as plt
import numpy as np

from ..Srb_With_Memory import mem_qubit_reset, srb_with_memory


def plot_return_prob_varying_error_rate(
    num_qubits,
    mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
    mem_err_func=mem_qubit_reset,
    num_samples=1000,
    reg_b_copies=1,
    correction_on_reg_b=True,
    filename=None,
    show_plot=False,
):
    """Find the survival after two gates are applied in example II.1 with different
    error rate.

    Args:
        num_qubits (int):
            Number of qubits to be characterized in the RB experiment. Defaults to 1.
        num_samples (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param_vals (np.array(int), optional):
            Controls the severity of the error. Plot as for each value given.
        mem_err_func (function):
            Error function to be applied to the memory
        reg_b_copies (int):
            Number of copies of register b used as a memory.
        correction_on_reg_b (bool):
            Whether or not to use error correction on register b.
        filename (directory):
            Name of file to save results in.
        show_plot (bool):
            If true, then run matplotlib.pyplot.show as soon as plot is created.

    """

    results = []
    for delta in mem_err_param_vals:
        results_2 = []
        print(delta)
        for _ in range(num_samples):
            results_2.append(
                srb_with_memory(
                    2,
                    num_qubits,
                    delta,
                    mem_err_func,
                    reg_b_copies=reg_b_copies,
                    correction_on_reg_b=correction_on_reg_b,
                )
            )
        results.append(1 - np.mean(results_2))

    plt.plot(mem_err_param_vals, results, marker="o", markersize=5)
    plt.ylabel("Probability of returning to |0> state")
    plt.xlabel("Probability of Reset")
    plt.title("How Probable is Returning to the |0> state")
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "_"
            + "survival_after_two_gates_"
            + str(num_qubits)
            + "_qubits"
        )
    plt.savefig("plots/" + filename + ".png")
    if show_plot:
        plt.show()
    plt.clf()

    return list(zip(mem_err_param_vals, results))


def plot_return_prob_varying_qubit_number(
    num_qubits_list,
    mem_err_param=0.05,
    mem_err_func=mem_qubit_reset,
    correction_on_reg_b=True,
    num_samples=1000,
    reg_b_copies=1,
    filename=None,
    show_plot=False,
):
    """Find the survival after two gates are applied in example II.1 with different
    error rate.

    Args:
        num_qubits_list (int):
            Numbers of qubits to test return probability for. Defaults to 1.
        num_samples (int, optional):
            Number of times the expectation value is collected from RB experiment.
            Defaults to 1000.
        mem_err_param (int, optional):
            Controls the severity of the error on the
        mem_err_func (function):
            Error function to be applied to the memory
        reg_b_copies (int):
            Number of copies of register b used as a memory.
        correction_on_reg_b (bool):
            Whether or not to use error correction on register b.
        filename (directory):
            Name of file to save results in.
        show_plot (bool):
            If true, then run matplotlib.pyplot.show as soon as plot is created.

    """

    results = []
    for num_qubits in num_qubits_list:
        results_2 = []
        print(num_qubits)
        for _ in range(num_samples):
            results_2.append(
                srb_with_memory(
                    2,
                    num_qubits,
                    mem_err_param,
                    mem_err_func,
                    reg_b_copies=reg_b_copies,
                    correction_on_reg_b=correction_on_reg_b,
                )
            )
        results.append(1 - np.mean(results_2))

    plt.plot(num_qubits_list, results, marker="o", markersize=5)
    plt.ylabel("Probability of returning to |0> state")
    plt.xlabel("Number of Qubits")
    plt.title("How Probable is Returning to the |0> state")
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "_"
            + "survival_after_two_gates_"
            + str(mem_err_param)
            + "memory_error_prob"
        )
    plt.savefig("plots/" + filename + ".png")
    if show_plot:
        plt.show()
    plt.clf()

    return list(zip(num_qubits_list, results))
