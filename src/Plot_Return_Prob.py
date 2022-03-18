import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from Srb_With_Memory import mem_qubit_reset, srb_with_memory

def plot_return_prob_varying_error_rate(num_qubits, mem_err_param_vals, mem_err_func, num_samples, reg_b_copies, correction_on_reg_b, filename, show_plot):
# def plot_return_prob_varying_error_rate(
#     num_qubits,
#     mem_err_param_vals=np.arange(0.0, 0.2, 0.02),
#     mem_err_func=mem_qubit_reset,
#     num_samples=1000,
#     reg_b_copies=1,
#     correction_on_reg_b=True,
#     filename=None,
#     show_plot=False,
# ):
	"""Find the survival after two gates are applied in example II.1 with different
	error rate.

	Args:
		num_qubits (int):
			Number of qubits to be characterized in the RB experiment. Defaults to 1.
		num_samples (int, optional):
			Number of times the expectation value is collected from RB experiment.
			Defaults to 1000.
		mem_err_param_vals (np.array(int), optional):
			Controls the severity of the error. Plot as for each value given.
		mem_err_func (function):
			Error function to be applied to the memory
		reg_b_copies (int):
			Number of copies of register b used as a memory.
		correction_on_reg_b (bool):
			Whether or not to use error correction on register b.
		filename (directory):
			Name of file to save results in.
		show_plot (bool):
			If true, then run matplotlib.pyplot.show as soon as plot is created.

	"""
	if filename is None:
		filename = (
			mem_err_func.__name__
			+ "_"
			+ "survival_after_two_gates_"
			+ str(num_qubits)
			+ "_qubits"
		)
	results = []
	data_file = "./../data/%s.npy" % (filename)
	if not os.path.isfile(data_file):
		for delta in mem_err_param_vals:
			delta_data_file = "./../data/%s_delta_%g.npy" % (filename, delta)
			results.append(np.load(delta_data_file))
		np.save(data_file, np.array(results))
	else:
		results = np.load(data_file)
	
	fig = plt.figure(figsize=(10, 8))
	plt.plot(mem_err_param_vals, results, marker="o", markersize=5, linewidth = 2)
	plt.ylabel("Probability of returning to $|0\\rangle$ state", fontsize = 18, labelpad = 10)
	plt.xlabel("Probability of Reset", fontsize = 18, labelpad = 10)
	# plt.title("How Probable is Returning to the |0> state")
	plt.tick_params(which="both", direction = "inout", labelsize = 18, pad = 5)
	plt.savefig("./../plots/" + filename + ".png", dpi = 300)
	if show_plot:
		plt.show()
	plt.clf()
	return list(zip(mem_err_param_vals, results))

def prob_varying_error_rate(core, num_qubits, delta, mem_err_func, num_samples, reg_b_copies, correction_on_reg_b, filename, show_plot):
	"""Find the survival after two gates are applied in example II.1 with different
	error rate.

	Args:
		num_qubits (int):
			Number of qubits to be characterized in the RB experiment. Defaults to 1.
		num_samples (int, optional):
			Number of times the expectation value is collected from RB experiment.
			Defaults to 1000.
		mem_err_param_vals (np.array(int), optional):
			Controls the severity of the error. Plot as for each value given.
		mem_err_func (function):
			Error function to be applied to the memory
		reg_b_copies (int):
			Number of copies of register b used as a memory.
		correction_on_reg_b (bool):
			Whether or not to use error correction on register b.
		filename (directory):
			Name of file to save results in.
		show_plot (bool):
			If true, then run matplotlib.pyplot.show as soon as plot is created.

	"""
	if filename is None:
		filename = (
			mem_err_func.__name__
			+ "_"
			+ "survival_after_two_gates_"
			+ str(num_qubits)
			+ "_qubits"
		)
	data_file = "./../data/%s_delta_%g.npy" % (filename, delta)
	if (not os.path.isfile(data_file)):
		results = []
		# for delta in mem_err_param_vals:
		# results_2 = []
		for __ in tqdm(range(num_samples), desc="delta = %g" % (delta), position=core):
			results.append(
				srb_with_memory(
					2,
					num_qubits,
					delta,
					mem_err_func,
					reg_b_copies=reg_b_copies,
					correction_on_reg_b=correction_on_reg_b,
				)
			)
		np.save(data_file, np.mean(results))
	return None


def prob_varying_qubit_number(core, nqubits, mem_err_param, mem_err_func, correction_on_reg_b, num_samples, reg_b_copies, filename, show_plot):
	"""Find the survival after two gates are applied in example II.1 with different
	error rate.

	Args:
		nqubits (int):
			Numbers of qubits to test return probability for. Defaults to 1.
		num_samples (int, optional):
			Number of times the expectation value is collected from RB experiment.
			Defaults to 1000.
		mem_err_param (int, optional):
			Controls the severity of the error on the
		mem_err_func (function):
			Error function to be applied to the memory
		reg_b_copies (int):
			Number of copies of register b used as a memory.
		correction_on_reg_b (bool):
			Whether or not to use error correction on register b.
		filename (directory):
			Name of file to save results in.
		show_plot (bool):
			If true, then run matplotlib.pyplot.show as soon as plot is created.

	"""
	data_file = "./../data/%s_nq_%d.npy" % (filename, nqubits)
	if (not os.path.isfile(data_file)):
		results = np.zeros(num_samples, dtype = np.double)
		for s in tqdm(range(num_samples), desc="nqubits = %d" % (nqubits), position = core):
			# print("num_qubits = {}".format(nqubits))
			prob = srb_with_memory(nqubits, 2, mem_err_param, mem_err_func, reg_b_copies=reg_b_copies, correction_on_reg_b=correction_on_reg_b)
			results[s] = prob
		np.save(data_file, 1 - np.mean(results))
	return None


def plot_return_prob_varying_qubit_number(
	num_qubits_list,
	mem_err_param=0.05,
	mem_err_func=mem_qubit_reset,
	correction_on_reg_b=True,
	num_samples=1000,
	reg_b_copies=1,
	filename=None,
	show_plot=False,
):
	"""Find the survival after two gates are applied in example II.1 with different
	error rate.

	Args:
		num_qubits_list (int):
			Numbers of qubits to test return probability for. Defaults to 1.
		num_samples (int, optional):
			Number of times the expectation value is collected from RB experiment.
			Defaults to 1000.
		mem_err_param (int, optional):
			Controls the severity of the error on the
		mem_err_func (function):
			Error function to be applied to the memory
		reg_b_copies (int):
			Number of copies of register b used as a memory.
		correction_on_reg_b (bool):
			Whether or not to use error correction on register b.
		filename (directory):
			Name of file to save results in.
		show_plot (bool):
			If true, then run matplotlib.pyplot.show as soon as plot is created.

	"""
	if filename is None:
		filename = (
			mem_err_func.__name__
			+ "_"
			+ "survival_after_two_gates_"
			+ str(mem_err_param)
			+ "memory_error_prob"
		)

	data_file = "./../data/%s.npy" % (filename)
	if (not os.path.isfile(data_file)):
		results = []
		for num_qubits in num_qubits_list:
			nqubits_data_file = "./../data/%s_nq_%d.npy" % (filename, num_qubits)
			results_2 = np.load(nqubits_data_file)
			results.append(results_2)
		np.save(data_file, np.array(results))
	
	results = np.load(data_file)

	fig = plt.figure(figsize=(10, 8))
	plt.plot(num_qubits_list, results, marker="o", markersize=10, linewidth=2) # Include error bars: p * (1 - p) / sqrt(N).
	plt.ylabel("Probability of returning to $|0\\rangle$ state", fontsize = 18, labelpad = 10)
	plt.xlabel("Number of Qubits", fontsize = 18, labelpad = 7)
	# plt.title("How Probable is Returning to the |0> state")
	# plt.xticks(np.linspace(0, max_seq_length - 1, 20), rotation=45)
	plt.tick_params(direction = "inout", which = "both", length = 8, labelsize = 18, pad = 5, width=1.5)
	
	plt.savefig("./../plots/" + filename + ".png", dpi = 300)
	if show_plot:
		plt.show()
	plt.clf()

	return list(zip(num_qubits_list, results))
