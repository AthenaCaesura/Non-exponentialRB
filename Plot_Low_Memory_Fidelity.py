# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:56:54 2021

@author: Athena

Runs and Plots results for low levels of memory fidelity in register B.
"""
from PresentRBResults import plot_shots

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for no_mem_error in [.4, .2, .1]:
            print("Current qubit number is: " + str(num_qubits))
            print("Current memory fidelity is: "
				  + str(100 * mem_fidelity) + "%")
            plot_shots(maxm=20, num_qubits=num_qubits, mem_fidelity=mem_fidelity)