from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from srb import srb_memory


def test_srb_memory():
    for _ in range(100):
        stuff = [srb_memory(100, 5, 0, lambda x, y: y) for _ in range(1)]


def test_scaling():
    times = np.array([])
    xrange = range(1, 20, 3)
    for i in xrange:
        print(i)
        start = time()
        srb_memory(10, i, 0, lambda x, y: y)
        times = np.append(times, [time() - start])

    xrange = np.array(xrange)

    def func(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(func, np.log10(xrange), np.log10(times))

    breakpoint()

    # Make plot of the scaling
    # plt.plot(np.log10(xrange), np.log10(times), "b-", label="data")
    # plt.plot(
    #     np.log10(xrange),
    #     func(np.log10(xrange), *popt),
    #     "r--",
    #     label="fit: m=%5.3f, b=%5.3f" % tuple(popt),
    # )
    # plt.ylabel("log(time to compute sequence of length 10)")
    # plt.xlabel("log(number of qubits)")
    # plt.title("How does time scale with number of qubits?")
    # plt.legend(loc="best")
    # plt.show()
