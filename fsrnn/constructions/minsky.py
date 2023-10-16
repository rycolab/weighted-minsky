from itertools import product

import numpy as np

from fsrnn.base.fsa import FSA
from fsrnn.base.semiring import Real
from fsrnn.base.state import State
from fsrnn.base.symbol import EOS, Sym
from fsrnn.models.elman_network import ElmanNetwork


class MinskyTransform:
    def __init__(self, A: FSA, projection: str = "softmax") -> None:
        """The class performing the Minsky transformation of an FSA into an Heaviside
        Elman RNN.

        Args:
            A (FSA): The FSA to be transformed.
            projection (str): What kind of projection from the scores to the normalized
                local probability distribution to use.
                Can be either "softmax" or "sparsemax", where "sparsemax" can only be
                used if the FSA is probabilistic.
                Defaults to "softmax".
        """
        assert A.deterministic, "The automaton must be deterministic."
        assert A.R == Real

        self.A = A
        self.Sigma = list(self.A.Sigma)
        self.SigmaEOS = self.Sigma + [EOS]
        self.n_states, self.n_symbols = len(self.A.Q), len(self.Sigma)
        self.D = self.n_states * (self.n_symbols + 1)

        self.construct()

    def hidden_state(self, state: State, symbol: Sym) -> np.ndarray:
        """Creates a one-hot encoding of the state-symbol pair to be used as the hidden
        state of the RNN.

        Args:
            state (State): Current state of the automaton.
            symbol (Sym): The symbol used to enter the state.

        Returns:
            np.ndarray: The one-hot encoding of the state-symbol pair.
        """
        h = np.zeros((self.D))
        h[self.n[(state, symbol)]] = 1
        return h

    def set_up_orderings(self):
        # The maps mapping the state-symbols pairs to their indices in the one-hot
        # encoding of the state-symbols pairs for the multiplication with the matrix U.
        # f is a map from (state, symbol) pairs to a their ids
        self.n = dict()
        self.n_inv = dict()
        for i, (q, a) in enumerate(product(self.A.Q, self.SigmaEOS)):
            self.n[(q, a)] = i
            self.n_inv[i] = (q, a)

        # The map mapping the symbols of the alphabet to their indices in the one-hot
        # encoding of the symbols for the multiplication with the matrix V.
        self.m = {a: i for i, a in enumerate(self.SigmaEOS)}

    def sym_one_hot(self, symbol):
        y = np.zeros((self.n_symbols + 1))
        y[self.m[symbol]] = 1
        return y

    def construct(self):
        self.set_up_orderings()

        self.U = np.zeros((self.D, self.D))
        self.V = np.zeros((self.D, self.n_symbols + 1))

        # make U
        # The matrix U will contain entries of 1 in the positions [f[(qʼ, b)], f[(q, a)]
        # where q - b -> qʼ is an arc in the FSA A.
        # The symbol a, on the other hand, is free, meaning that the number of ones in
        # every row of U is |Sigma| + 1 and the number of ones in every column of U is
        # at most |Sigma| + 1 (one for each symbol in the alphabet and one for EOS).
        # This encodes that, after arriving into the state q with the symbol a, we can
        # transition into any of the states q with the symbol b.
        for q, a in product(self.A.Q, self.Sigma):
            for b, qʼ, _ in self.A.arcs(q):
                self.U[self.n[(qʼ, b)], self.n[(q, a)]] = 1

        # make V
        # The matrix V will contain entries of 1 in the positions [f[(qʼ, a)], g[a]],
        # where q - a -> qʼ is an arc in the FSA A for some q in Q.
        # This just means that qʼ has an incoming transition with the symbol a.
        for q in self.A.Q:
            for a, qʼ, _ in self.A.arcs(q):
                self.V[self.n[(qʼ, a)], self.m[a]] = 1

        self.R = self.rnn()

    def rnn(self) -> ElmanNetwork:
        E = -np.infty * np.ones((self.n_symbols + 1, self.D))

        for p, a in product(self.A.Q, self.Sigma):
            for b, _, w in self.A.arcs(p):
                E[self.m[b], self.n[(p, a)]] = np.log(w.value)

        for a in self.Sigma:
            for p, w in self.A.F:
                # The final weight is an alternative "output" weight
                # for the final states.
                E[self.m[EOS], self.n[(p, a)]] = np.log(w.value)

        q0 = list(self.A.I)[0][0]

        return ElmanNetwork(
            Sigma=self.Sigma,
            U=self.U,
            V=self.V,
            b=-1 * np.ones(shape=(self.D)),
            E=E,
            h0=self.hidden_state(q0, self.Sigma[0]),
            f=lambda x: x,  # TODO
            one_hot=lambda s: self.sym_one_hot(s),
        )
