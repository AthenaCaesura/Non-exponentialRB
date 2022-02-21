from time import time

import numpy as np
import pytest
from NonExponentialRB.Srb_With_Memory import mem_qubit_reset, srb_with_memory
from scipy.optimize import curve_fit


@pytest.mark.parametrize(
    "num_qubits",
    [1, 2, 5],
)
def test_srb_with_memory_no_error(num_qubits):
    max_seq_length = 50
    target = [0, 1] * (max_seq_length // 2)
    samples = [
        srb_with_memory(
            seq_length,
            num_qubits,
            0,
            mem_qubit_reset,
        )
        for seq_length in range(max_seq_length)
    ]
    assert target == samples


@pytest.mark.parametrize(
    "num_qubits",
    [1, 5],
)
def test_srb_with_memory_with_error(num_qubits):
    max_seq_length = 50
    target = [0, 1] * (max_seq_length // 2)
    samples = [
        srb_with_memory(
            seq_length,
            num_qubits,
            0.1,
            mem_qubit_reset,
        )
        for seq_length in range(max_seq_length)
    ]
    assert target != samples


def test_scaling():
    times = np.array([])
    xrange = range(1, 30, 3)
    num_samples = 10
    for i in xrange:
        print(i)
        for _ in range(num_samples):
            start = time()
            srb_with_memory(10, i, 0, lambda x, y: y)
            times = np.append(times, [time() - start])

    xrange = np.array(xrange)

    def func(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(
        func, np.log10([i for i in xrange for _ in range(num_samples)]), np.log10(times)
    )

    # for some reason we get excellent scaling here, O(num_qubits^2) in time.
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
