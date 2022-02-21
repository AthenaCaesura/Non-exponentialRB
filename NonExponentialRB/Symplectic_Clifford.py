import numpy as np

from .Invert_Binary_Matrix import gf2elim


class SymplecticClifford:
    """Class for representing a clifford using symplectic matrices and paulis.

    Attributes:
        num_qubits (int):
            Number of qubits of the whole space the clifford is acting on.
        table (np.ndarray):
            Array representing the clifford of the form
            (sympletic matrix | pauli vector)
        global_phase (float):
            Global phase of the Clifford.
    """

    def __init__(self, symp, global_phase=1):
        symp = np.array(symp, dtype=int)
        n = len(symp) // 2
        if 2 * n != len(symp[0]) - 1 or len(symp) % 2 != 0:
            raise ValueError(f"Size {symp.shape} is not a valid Clifford Operation.")
        self.num_qubits = n
        self.table = symp
        self.global_phase = global_phase

    def __mul__(self, other):
        new_table = np.dot(self.table[:, :-1], other.table) % 2
        new_table[:, -1] = np.array(
            [i ^ j for i, j in zip(new_table[:, -1], self.table[:, -1])]
        )
        return SymplecticClifford(new_table, self.global_phase * other.global_phase)

    def inv(self):
        """Makes self the inverse SymplecticClifford to itself."""
        augmented_matrix = np.column_stack(
            (self.table[:, :-1], np.eye(2 * self.num_qubits, dtype=int))
        )
        symp_inverse = gf2elim(augmented_matrix)[:, 2 * self.num_qubits :]
        self.table = np.column_stack(
            (
                symp_inverse,
                np.dot(symp_inverse, self.table[:, -1]) % 2,
            )
        )

    def measure(self):
        """Assuming that the symplectic part is identity, measure overlap
        with |00..0> state. With these assumptions, we only need to look at if
        there are any X's in the pauli vector. O(num_qubits) time."""
        assert np.array_equal(self.table[:, :-1], np.eye(2 * self.num_qubits))
        return 1 if 1 in self.table[1::2, -1] else 0

    def evolve_pauli(self, pauli: np.ndarray):
        """Evolves a given pauli by the clifford represented by self.
            Ignores phases imparted on the pauli by the clifford.

        Args:
            pauli (np.array): Pauli represented in typical tableau
            format omitting phase.
        """
        return np.dot(self.table[:, :-1], pauli) % 2

    def pauli_mult(self, pauli):
        """Quickly multiplies self.table by a pauli in O(num_qubits) time.

        Args:
            pauli (np.array): array encoding a pauli
        """
        new_pauli = [i ^ j for i, j in zip(self.table[:, -1], pauli)]
        new_table = np.column_stack((self.table[:, :-1], new_pauli))
        return SymplecticClifford(new_table, self.global_phase)


def assert_commutations(table):
    """Test if given table corresponds to proper commutation relations that
    a matrix in the chp representation.

    Args:
        table (np.ndarray): Table to test commutation relations for.
    """
    num_qubits = len(table) // 2

    target = np.array(
        [
            [1 if i == j + num_qubits else 0 for j in range(num_qubits * 2)]
            for i in range(num_qubits * 2)
        ]
    )
    comm_mat = np.zeros((2 * num_qubits, 2 * num_qubits))
    for i in range(2 * num_qubits):
        for j in range(i):
            comm_mat[i, j] = symplectic_inner_product(table[i, :-1], table[j, :-1])
    assert np.linalg.det(table[:, :-1]) != 0
    assert np.array_equal(comm_mat, target)


def symplectic_inner_product(vec_1: np.ndarray, vec_2: np.ndarray) -> bool:
    """Symplectic inner product of vec_1 and vec_2.

    Args:
        row_i (np.ndarray): (1 x 2 * num_qubits) Binary vector repsenting a pauli.
        row_j (np.ndarray): (1 x 2 * num_qubits) Binary vector repsenting a pauli.

    Returns: int
        Symplectic inner product of vectors.
    """
    if len(vec_1) % 2 != 0 or len(vec_1) != len(vec_2):
        raise TypeError("Vectors must be same size and of form 2*num_qubits")

    n = len(vec_1) // 2
    prod = np.dot(vec_1[:n], vec_2[n : 2 * n])
    prod += np.dot(vec_2[:n], vec_1[n : 2 * n])
    prod %= 2
    return prod
