import matplotlib.pyplot as plt
import numpy as np

from Srb_With_Memory import mem_qubit_reset, srb_with_memory


def plot_return_prob_varying_error_rate(
    num_qubits,
    mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
    mem_err_func=mem_qubit_reset,
    reg_b_copies=1,
    num_samples=1000,
    filename=None,
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
        filename (directory):
            Name of file to save results in.

    """

    results = []
    for delta in mem_err_param_vals:
        results_2 = []
        for _ in range(1000):
            results_2.append(
                srb_with_memory(
                    num_qubits,
                    maxm=2,
                    num_samples=num_samples,
                    mem_err_param=delta,
                    mem_err_func=mem_err_func,
                    reg_b_copies=reg_b_copies,
                )
            )
        results.append(1 - np.mean(results_2))

    plt.plot(mem_err_param_vals, results)
    plt.ylabel("Probability of returning to |0> state")
    plt.xlabel("Probability of Reset")
    plt.title("How Probable is Returning to the |0> state")
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "/"
            + "survival_after_two_gates_"
            + str(num_qubits)
            + "_qubits"
        )
    plt.savefig("plots/" + filename + ".png")
    plt.clf()
    plt.show()

    return zip(mem_err_param_vals, results)


def plot_return_prob_varying_qubit_number(
    num_qubits_list,
    mem_err_param=0.05,
    mem_err_func=mem_qubit_reset,
    reg_b_copies=1,
    num_samples=1000,
    filename=None,
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
        filename (directory):
            Name of file to save results in.

    """

    results = []
    for num_qubits in num_qubits_list:
        results_2 = []
        for _ in range(1000):
            results_2.append(
                srb_with_memory(
                    num_qubits,
                    maxm=2,
                    num_samples=num_samples,
                    mem_err_param=mem_err_param,
                    mem_err_func=mem_err_func,
                    reg_b_copies=reg_b_copies,
                )
            )
        results.append(1 - np.mean(results_2))

    plt.plot(num_qubits_list, results)
    plt.ylabel("Probability of returning to |0> state")
    plt.xlabel("Number of Qubits")
    plt.title("How Probable is Returning to the |0> state")
    if filename is None:
        filename = (
            mem_err_func.__name__
            + "/"
            + "survival_after_two_gates_"
            + str(num_qubits)
            + "_qubits"
        )
    plt.savefig("plots/" + filename + ".png")
    plt.clf()
    plt.show()

    return zip(num_qubits_list, results)
