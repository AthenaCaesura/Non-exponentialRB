"""See Figure_plan.md to see specifications of plots
"""

from ..Plotting.Plot_Expectation_Values import plot_expectation_values
from ..Srb_With_Memory import mem_qubit_reset


def main():
    for num_qubits, reg_b_copies in zip([1, 5, 20], [10, 150, 2000]):
        evals = plot_expectation_values(
            num_qubits,
            max_seq_length=20,
            mem_err_param=0.95,
            mem_err_func=mem_qubit_reset,
            num_samples=1000,
            reg_b_copies=reg_b_copies,
            correction_on_reg_b=False,
            filename="figures/figure_4",
        )
        # Print evalss for confirmation
        print(evals)
