from random import sample
from math import sqrt
from numpy import kron, eye
from utils import H, P, CNOT
import numpy as np

# =============================================================================
# We implement the algorithm given in GENERATING RANDOM ELEMENTS OF A
# FINITE GROUP (Celler et. al.) which involves taking the generating set
# and applying elements randomly until the uniform distribution is obtained.
#
# S is the central database from which random elements are sampled.
#
# In order to use properly, one must sample from a large enough array,
# for most uses n=3 is fine but it may need to be adjusted for larger
# groups see (Celler et. al.) for details.
# This system allows one to define multiplication of elements in the
# group uniquely for each sampling. Default multiplication is "*"
#
# NOTE: The group product MUST be robust to many multiplications i.e. the group
# is always closed.
# =============================================================================


class SampleGroup(object):

    def __init__(self, gens, n=3, Multiply=lambda a, b: a * b):
        """
        Initializes self.S so that random elements can be drawn from it.

        Parameters
        ----------
        gens : Usually a Matrix
            Generators of the group to be sampled
        n : int, optional
            number of copies of gens in self.S. also initializes size of
                self.S. The default is 3.
        Multiply : TYPE, optional
            define multiplication for the group. The default is "8".

        Returns
        -------
        None.

        """
        self.S = [gen for gen in gens for _ in range(n)]
        self.Multiply = Multiply
        self.randomize()

    def shuffle(self, vals=None):
        """
        Multiply one element of self.S by another and replace it in self.S

        Parameters
        ----------
        vals : list, optional
            2 integer list specifying the elements in self.S to be multiplied,
            with the resulting element replacing the first element in self.S

        Returns
        -------
        None.

        """
        if(vals is None):
            vals = sample(range(len(self.S)), 2)
        self.S[vals[0]] = self.Multiply(self.S[vals[0]], self.S[vals[1]])
        self.S[vals[0]] = np.round(self.S[vals[0]], 8)

    def randomize(self, k=100):
        """
        Make self.S full of random elements of the group to be sampled

        Parameters
        ----------
        k : int, default 100
            number of times we multiply a group element by another and
                replace in self.S

        Returns
        -------
        None.

        """
        for i in range(k):
            self.shuffle()

    def sample(self):
        """
        Get a random element from the group

        Returns
        -------
        Group Element
            random element of the group

        """
        vals = sample(range(len(self.S)), 2)
        self.shuffle(vals)
        return self.S[vals[1]]


class CliffordSampler(SampleGroup):
    def __init__(self, numqubits=1):
        if numqubits == 1:
            gens = np.array([H, P])
        else:
            gens = np.array([kron(H, eye(2**(numqubits - 1))),
                             kron(P, eye(2**(numqubits - 1))),
                             kron(CNOT, eye(2**(numqubits - 2)))])
        super().__init__(gens, n=3,
                         Multiply=lambda a, b: np.matmul(a, b))
