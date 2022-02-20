"""See Figure_plan.md to see specifications of test
"""

import numpy as np

from Plot_Return_Prob import (
    plot_return_prob_varying_error_rate,
    plot_return_prob_varying_qubit_number,
)
from Srb_With_Memory import mem_qubit_reset


def __main__():
    # Plot part a of figure 3
    results = plot_return_prob_varying_error_rate(
        5,
        mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
        mem_err_func=mem_qubit_reset,
        reg_b_copies=1,
        num_samples=1000,
        filename=None,
        filename="figures/figure_3_a.png",
    )

    print(results)

    # Plot part b of figure 3
    results = plot_return_prob_varying_qubit_number(
        list(range(1, 10, 2)),
        mem_err_param_vals=0.02,
        mem_err_func=mem_qubit_reset,
        reg_b_copies=1,
        num_samples=1000,
        filename=None,
        filename="figures/figure_3_b.png",
    )

    print(results)
