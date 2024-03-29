"""See Figure_plan.md to see specifications of plot.

Should finish in around ~5 mins on a laptop.
"""
from Plot_Expectation_Values import plot_expectation_values
from Srb_With_Memory import mem_qubit_reset


if __name__ == '__main__':
    evals = plot_expectation_values(
        5,
        max_seq_length=20,
        mem_err_param=0,
        mem_err_func=mem_qubit_reset,
        num_samples=1000,
        filename="figure_1",
    )
    # Print expectation values for confirmation
    print(evals)
