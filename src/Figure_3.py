"""See Figure_plan.md to see specifications of plots

This file might take about ~20 mins to run on a laptop
"""

import numpy as np

from Plot_Return_Prob import prob_varying_error_rate, plot_return_prob_varying_error_rate, prob_varying_qubit_number, plot_return_prob_varying_qubit_number
from Srb_With_Memory import mem_qubit_reset, srb_with_memory

import multiprocessing as mp

if __name__ == '__main__':
	# Plot part a of figure 3
	mem_err_param_vals=np.arange(0.0, 0.2, 0.02)
	n_complete = 0
	while (n_complete < mem_err_param_vals.size):
		ncores = min(mp.cpu_count(), mem_err_param_vals.size - n_complete)
		processes = []
		for p in range(ncores):
			# num_qubits, mem_err_param_vals, mem_err_func, num_samples, reg_b_copies, correction_on_reg_b, filename, show_plot
			# processes.append(mp.Process(target=prob_varying_error_rate, args=(p, 5, mem_err_param_vals[n_complete + p], mem_qubit_reset, 100000, 1, True, "figure_3_a", False)))
			plot_return_prob_varying_error_rate(5, mem_err_param_vals, mem_qubit_reset, 100000, 1, True, "figure_3_a", False)
		# for p in range(ncores):
		# 	processes[p].start()
		# for p in range(ncores):
		# 	processes[p].join()
		n_complete += ncores

	# Plot data.
	plot_return_prob_varying_error_rate(5, mem_err_param_vals, mem_qubit_reset, 100000, 1, True, "figure_3_a", False)
	# results = plot_return_prob_varying_error_rate(
	# 	5,
	# 	mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
	# 	mem_err_func=mem_qubit_reset,
	# 	reg_b_copies=1,
	# 	num_samples=100000,
	# 	filename="figure_3_a",
	# 	0
	# )

	# print(results)

	# Plot part b of figure 3
	num_qubits = np.arange(1, 10, 2, dtype = np.int32)
	n_complete = 0
	while (n_complete < num_qubits.size):
		ncores = min(mp.cpu_count(), num_qubits.size - n_complete)
		processes = []
		for p in range(ncores):
			# processes.append(mp.Process(target=prob_varying_qubit_number, args=(p, num_qubits[n_complete + p], 0.02, mem_qubit_reset, 10000, 1, True, "figure_3_b", False)))
			prob_varying_qubit_number(p, num_qubits[n_complete + p], 0.02, mem_qubit_reset, 1, 10000, True, "figure_3_b", False)
		# for p in range(ncores):
		# 	processes[p].start()
		# for p in range(ncores):
		# 	processes[p].join()
		n_complete += ncores

	results = plot_return_prob_varying_qubit_number(num_qubits, 0.02, mem_qubit_reset, 10000, 1, True, "figure_3_b", False)

	# print(results)
