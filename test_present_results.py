from time import time

import numpy as np
from scipy.optimize import curve_fit

from PresentRBResults import plot_shots


def test_scaling():
    times = np.array([])
    xrange = range(2, 6)
    for i in xrange:
        print(f"testing {i} qubits")
        start = time()
        plot_shots(
            i,
            mem_err_param=0.95,
            reg_b_copies=150,
            filename="tests/test_with_EC_5",
            num_trials=100,
        )
        times = np.append(times, [time() - start])

    xrange = np.array(xrange)

    popt, pcov = curve_fit(
        linear_model, np.log10([i for i in xrange for _ in range(1)]), np.log10(times)
    )

    print(f"scales like O(n^({popt[0]}) with constant overhead {popt[1]}\n")

    print("Run time with 150 copies in register b and 1000 shots.\n")
    print(" qubits | time (hours) ")
    print("--------|--------------")
    for i in range(1, 20):
        print(f"    {i}   | {run_time_estimator(i, popt[0], popt[1])}")


def linear_model(x, m, b):
    return m * x + b


def run_time_estimator(num_qubits, m, b):
    time_for_100_shots = 10 ** linear_model(np.log10(num_qubits), m, b)
    # assume linear scaling in number of shots
    time_for_1000_shots = time_for_100_shots * 10
    # return time in hours
    return round(time_for_1000_shots / (60 ** 2), 1)


plot_shots(
    2,
    mem_err_param=0.0,
    reg_b_copies=3,
    filename="tests/test_with_EC_5",
    num_trials=1,
)

"""
Paper needs:
1. High error rate w/num qubits :1, 5, high as we can go
   Also, need high number of copies

"""
