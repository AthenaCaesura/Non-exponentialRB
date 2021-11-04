# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:56:54 2021

@author: Athena

Runs and Plots results for low levels of memory fidelity in register B.
"""
from PresentRBResults import plot_shots

if __name__ == "__main__":
    for num_qubits in [1, 3, 5]:
        for no_mem_err_prob in [.5, .25]:
            mem_err_prob = round(1-pow(no_mem_err_prob, 1/(2 * num_qubits)), 2)
            print("Current qubit number is: " + str(num_qubits))
            print("Current probability of a memory error is: "
				  + str(100 * mem_err_prob) + "%")
            plot_shots(num_qubits=num_qubits, mem_err_prob=mem_err_prob)