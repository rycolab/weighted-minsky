from typing import DefaultDict, Union

import numpy as np
from frozendict import frozendict

from fsrnn.base.semiring import Semiring
from fsrnn.base.state import State


class Strategy:
    VITERBI = 1
    LEHMANN = 2

    @staticmethod
    def str2strategy(name: str) -> int:  # noqa: C901
        """Returns the strategy corresponding to the given name.

        Args:
            name (str): The name of the strategy.

        Returns:
            int: The corresponding strategy.
        """
        if name.lower() == "viterbi":
            return Strategy.VITERBI
        elif name.lower() == "lehmann":
            return Strategy.LEHMANN
        else:
            raise ValueError(f"Unknown strategy {name}")


class Pathsum:
    def __init__(self, fsa):
        # basic FSA stuff
        self.fsa = fsa
        self.R = fsa.R
        self.N = self.fsa.num_states

        # state dictionary
        self.I = {}  # noqa: E741
        for n, q in enumerate(self.fsa.Q):
            self.I[q] = n

        # lift into the semiring
        self.W = self.lift()

    def _convert(self):
        mat = np.zeros((self.N, self.N))
        for n in range(self.N):
            for m in range(self.N):
                mat[n, m] = self.W[n, m].value
        return mat

    def lift(self):
        """creates the weight matrix from the automaton"""
        W = self.R.zeros(self.N, self.N)
        for p in self.fsa.Q:
            for a, q, w in self.fsa.arcs(p):
                W[self.I[p], self.I[q]] += w
        return W

    def pathsum(self, strategy: Union[int, str]) -> Semiring:  # noqa: C901
        if isinstance(strategy, str):
            strategy = Strategy.str2strategy(strategy)

        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_pathsum()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_pathsum()

        else:
            raise NotImplementedError

    def forward(self, strategy):
        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_fwd()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_fwd()

        else:
            raise NotImplementedError

    def backward(self, strategy: int) -> DefaultDict[State, Semiring]:  # noqa: C901
        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_bwd()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_bwd()

        else:
            raise NotImplementedError

    def viterbi_pathsum(self):
        pathsum = self.R.zero
        β = self.viterbi_bwd()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * β[q]
        return pathsum

    def viterbi_bwd(self) -> DefaultDict[State, Semiring]:
        """The Viterbi algorithm run backwards"""

        assert self.fsa.acyclic

        # chart
        β = self.R.chart()

        # base case (paths of length 0)
        for q, w in self.fsa.F:
            β[q] = w

        # recursion
        for p in self.fsa.toposort(rev=True):
            for _, q, w in self.fsa.arcs(p):
                β[p] += w * β[q]

        return β

    def allpairs_bwd(self, W):
        β = self.R.chart()
        W = self.lehmann()
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                β[p] += W[p, q] * self.fsa.ρ[q]
        return frozendict(β)

    def allpairs_pathsum(self, W):
        pathsum = self.R.zero
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                pathsum += self.fsa.λ[p] * W[p, q] * self.fsa.ρ[q]
        return pathsum

    def _lehmann(self, zero=True):
        """
        Lehmann's (1977) algorithm.
        """

        # initialization
        V = self.W.copy()
        U = self.W.copy()

        # basic iteration
        for j in range(self.N):
            V, U = U, V
            V = self.R.zeros(self.N, self.N)
            for i in range(self.N):
                for k in range(self.N):
                    # i ➙ j ⇝ j ➙ k
                    V[i, k] = U[i, k] + U[i, j] * U[j, j].star() * U[j, k]

        # post-processing (paths of length zero)
        if zero:
            for i in range(self.N):
                V[i, i] += self.R.one

        return V

    def lehmann(self, zero=True):
        # TODO: check we if we can't do away with this method.

        V = self._lehmann(zero=zero)

        W = {}
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                if p in self.I and q in self.I:
                    W[p, q] = V[self.I[p], self.I[q]]
                elif p == q and zero:
                    W[p, q] = self.R.one
                else:
                    W[p, q] = self.R.zero

        return frozendict(W)

    def lehmann_pathsum(self):
        return self.allpairs_pathsum(self.lehmann())

    def lehmann_fwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_fwd(self.lehmann())

    def lehmann_bwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_bwd(self.lehmann())
