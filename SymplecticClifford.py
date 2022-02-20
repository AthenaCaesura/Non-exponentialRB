import numpy as np

from Invert_binary_matrix import gf2elim


class SymplecticClifford:
    def __init__(self, symp, global_phase=1):
        n = len(symp) // 2
        if 2 * n != len(symp[0]) - 1 or len(symp) % 2 != 0:
            raise ValueError(f"Size {symp.shape} is not a valid Clifford Operation.")
        self.num_qubits = n
        self.table = np.array(symp, dtype=int)
        self.global_phase = global_phase

    def __mul__(self, other):
        new_table = np.dot(self.table[:, :-1], other.table) % 2
        new_table[:, -1] = np.array(
            [i ^ j for i, j in zip(new_table[:, -1], self.table[:, -1])]
        )
        return SymplecticClifford(new_table, self.global_phase * other.global_phase)

    def inv(self):
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
