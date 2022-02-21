"""See Figure_plan.md to see specifications of plots

This file might take about ~20 mins to run on a laptop
"""

import numpy as np

from ..Plotting.Plot_Return_Prob import (
    plot_return_prob_varying_error_rate,
    plot_return_prob_varying_qubit_number,
)
from ..Srb_With_Memory import mem_qubit_reset


def main():
    # Plot part a of figure 3
    results = plot_return_prob_varying_error_rate(
        5,
        mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
        mem_err_func=mem_qubit_reset,
        reg_b_copies=1,
        num_samples=10000,
        filename="figures/figure_3_a",
    )

    print(results)

    # Plot part b of figure 3
    results = plot_return_prob_varying_qubit_number(
        list(range(1, 10, 2)),
        mem_err_param=0.02,
        mem_err_func=mem_qubit_reset,
        reg_b_copies=1,
        num_samples=10000,
        filename="figures/figure_3_b",
    )

    print(results)
