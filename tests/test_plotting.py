import os

from NonExponentialRB.Plotting.Plot_Expectation_Values import plot_expectation_values
from NonExponentialRB.Srb_With_Memory import mem_qubit_reset


def test_no_error_plot():
    if os.path.exists("plots/test_plots/no_error.png"):
        os.remove("plots/test_plots/no_error.png")

    target = [(i, float((i + 1) % 2)) for i in range(1, 20)]
    evals = plot_expectation_values(
        1,
        max_seq_length=20,
        mem_err_param=0,
        mem_err_func=mem_qubit_reset,
        num_samples=100,
        filename="test_plots/no_error",
    )

    assert os.path.exists("plots/test_plots/no_error.png")
    assert evals == target
