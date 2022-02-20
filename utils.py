import numpy as np


def symplectic_inner_product(vec_1: np.ndarray, vec_2: np.ndarray) -> bool:
    """Symplectic inner product of vec_1 and vec_2.

    Args:
        row_i (np.ndarray): (1 x 2 * num_qubits) Binary vector repsenting a pauli.
        row_j (np.ndarray): (1 x 2 * num_qubits) Binary vector repsenting a pauli.

    Returns: int
        Symplectic inner product of vectors.
    """
    if len(vec_1) % 2 == 0 or len(vec_1) != len(vec_2):
        raise TypeError("Vectors must be same size and of form 2*num_qubits")

    n = len(vec_1) // 2
    prod = np.dot(vec_1[:n], vec_2[n : 2 * n])
    prod += np.dot(vec_2[:n], vec_1[n : 2 * n])
    prod %= 2
    return prod
