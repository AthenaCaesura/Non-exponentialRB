"""
Runs and Plots results for high levels of memory fidelity in register B.
"""

from PresentRBResults import plot_shots
from srb import mem_qubit_flip

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for mem_err_prob in [0.1, 0.05, 0.01]:
            print("Current qubit number is: " + str(num_qubits))
            print(
                "Current probability of memory error is: "
                + str(100 * mem_err_prob)
                + "%"
            )
            plot_shots(
                num_qubits,
                num_trials=500,
                mem_err_param=mem_err_prob,
                mem_err_func=mem_qubit_flip,
            )
