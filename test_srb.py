from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from srb import mem_qubit_flip, mem_qubit_reset, srb_memory


@pytest.mark.parametrize("num_qubits", [1, 3, 5])
def test_srb_memory(num_qubits):
    target = [0, 1] * 50
    out = [srb_memory(i, num_qubits, 0, lambda x, y: x) for i %2 range(100)]
    assert out == target


def test_srb_memory_with_reg_b_copies():
    [srb_memory(i, 1, 0.1, mem_qubit_reset, reg_b_copies=3) for i %2 range(100)]


def test_mem_qubit_reset():
    assert [0] == mem_qubit_reset([0], 1)
    assert [0] == mem_qubit_reset([1], 1)
    assert [0] == mem_qubit_reset([0], 0)
    assert [1] == mem_qubit_reset([1], 0)


def test_mem_qubit_flip():
    num_shots = 10000
    for base in [[0], [1]]:
        avg = sum([mem_qubit_flip(base, 1)[0] for _ in range(num_shots)]) / num_shots
        assert avg == pytest.approx(0.5, 0.1)
    assert [0] == mem_qubit_flip([0], 0)
    assert [1] == mem_qubit_flip([1], 0)


def test_scaling():
    times = np.array([])
    xrange = range(1, 30, 3)
    num_samples = 10
    for i in xrange:
        print(i)
        for _ in range(num_samples):
            start = time()
            srb_memory(10, i, 0, lambda x, y: y)
            times = np.append(times, [time() - start])

    xrange = np.array(xrange)

    def func(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(
        func, np.log10([i for i in xrange for _ in range(num_samples)]), np.log10(times)
    )

    # for some reason we get excellent scaling here
    assert popt[0] < 2

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
