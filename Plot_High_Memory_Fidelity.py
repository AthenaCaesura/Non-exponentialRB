# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:54:14 2021

@author: Athena

Runs and Plots results for high levels of memory fidelity in register B.
"""

from PresentRBResults import plot_shots

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for mem_err_prob in [.9, .95, .99]:
            print("Current qubit number is: " + str(num_qubits))
            print("Current probability of a memory error is: "
				  + str(100 * mem_err_prob) + "%")
            plot_shots(num_qubits=num_qubits, mem_err_prob=mem_err_prob)