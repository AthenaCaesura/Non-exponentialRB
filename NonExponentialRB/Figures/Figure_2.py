"""See Figure_plan.md to see specifications of plots

Should finish in ~5 mins on a laptop.
"""

from ..Plotting.Plot_Expectation_Values import plot_expectation_values
from ..Srb_With_Memory import mem_qubit_reset


def main():
    evals = plot_expectation_values(
        5,
        max_seq_length=20,
        mem_err_param=0.02,
        mem_err_func=mem_qubit_reset,
        num_samples=100,
        samples_per_shot=10,
        correction_on_reg_b=False,
        filename="figures/figure_2",
    )
    # Print evals for confirmation
    print(evals)
