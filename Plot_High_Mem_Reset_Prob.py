"""
Runs and Plots results for low levels of memory fidelity in register B.
"""
from PresentRBResults import plot_shots
from srb import mem_qubit_reset

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for mem_err_prob in [0.6, 0.8, 0.9]:
            print("Current qubit number is: " + str(num_qubits))
            print(
                "Current probability of memory error is: "
                + str(100 * mem_err_prob)
                + "%"
            )
            plot_shots(
                num_qubits,
                maxm=20,
                mem_err_param=mem_err_prob,
                mem_err_func=mem_qubit_reset,
            )
