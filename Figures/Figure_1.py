"""See Figure_plan.md to see specifications of test
"""
from Plot_Expectation_Values import plot_expectation_values
from Srb_With_Memory import mem_qubit_reset


def __main__():
    evals = plot_expectation_values(
        5,
        maxm=20,
        mem_err_param=0,
        mem_err_func=mem_qubit_reset,
        num_samples=1000,
        filename="figures/figure_1.png",
    )
    # Print expectation values for confirmation
    print(evals)
