"""
Runs and Plots results for high levels of memory fidelity in register B.
"""

from PresentRBResults import plot_shots

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for mem_fidelity in [.9, .95, .99]:
            print("Current qubit number is: " + str(num_qubits))
            print("Current memory fidelity is: "
				  + str(100 * mem_fidelity) + "%")
            plot_shots(num_qubits=num_qubits, mem_fidelity=mem_fidelity)