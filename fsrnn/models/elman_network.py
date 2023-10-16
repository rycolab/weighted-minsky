from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp

from fsrnn.base.alphabet import Alphabet
from fsrnn.base.symbol import Sym
from fsrnn.models.activations import H
from fsrnn.models.utils import to_compatible_string


class ElmanNetwork:
    def __init__(
        self,
        Sigma: Alphabet,
        U: np.ndarray,
        V: np.ndarray,
        b: np.ndarray,
        E: np.ndarray,
        h0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        one_hot: Callable[Sym, np.ndarray],
        σ: Callable[[np.ndarray], np.ndarray] = H,
        n_applications: int = 1,
    ):
        """Implementation of an Elman network with the Heaviside non-linearity.

        DEFINITION
        A Elman language model with the Heaviside non-linearity is a tuple
        • Σ is an alphabet of symbols;
        • h0 is a D-dimensional initial vector;
        • U a D x D transition matrix;
        • V a D x |Σ| symbol matrix;
        • E a |Σ| x (D+1) emission matrix;
        • b a |Σ|-dimensional bias term.
        • f is a projection function from the unnormalized scores to the normalized
            local probability distribution over the next symbol.
        • σ is the hidden state activation function.

        Args:
            Sigma (Alphabet): The alphabet of symbols.
            U (np.ndarray): The transition matrix.
            V (np.ndarray): The symbol matrix.
            b (np.ndarray): The bias term.
            E (np.ndarray): The emission matrix.
            h0 (np.ndarray): The initial hidden state.
            f (Callable[[np.ndarray], np.ndarray]): The projection function.
            σ (Callable[[np.ndarray], np.ndarray], optional): The hidden state
                activation function. Defaults to H (Heaviside).
            one_hot (Callable[Sym, np.ndarray]): The function that maps a symbol to its
                one-hot encoding.
            n_applications (int, optional): The number of times the Elman update step
                is applied to the hidden state.
        """

        self.Sigma = Sigma

        self.U = U
        self.V = V
        self.b = b
        self.E = E
        self.h0 = h0
        self.f = f
        self.sym_one_hot = one_hot
        self.σ = σ
        self.n_applications = n_applications

    # TODO: Make this work with arbitrary projection function
    def score(self, s: str) -> float:
        # Here, we assume that the special input symbol BOS has already been used
        # to arrive to the state encoded in h0.
        # This means that the function `to_compatible_string` does not append
        # the BOS symbol to the string.
        s = to_compatible_string(s)

        logp = 0.0  # Starting weight in a language model is 1.0
        h = self.h0

        for a in s:
            y = self.sym_one_hot(a)
            h, logp_ = self(h, y)
            logp += logp_

        return logp

    def __call__(
        self, h: np.ndarray, y: Union[str, Sym, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(y, str):
            y = Sym(y)

        if isinstance(y, Sym):
            y = self.sym_one_hot(y)

        logits = self.E[:, h.argmax()]  # logits = Eh

        # This is a way to avoid the issue that inf * 0 = nan.
        logp = logits[y.argmax()] - logsumexp(logits)

        for ll in range(self.n_applications):
            h = self.σ(np.dot(self.U, h) + np.dot(self.V, y) + self.b)

        return h, logp


class ElmanNetworkV2:
    def __init__(
        self,
        Sigma: Alphabet,
        U: np.ndarray,
        V: np.ndarray,
        b: np.ndarray,
        W: np.ndarray,
        E: np.ndarray,
        h0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        one_hot: Callable[Sym, np.ndarray],
        σ: Callable[[np.ndarray], np.ndarray] = H,
        n_applications: int = 1,
    ):
        """Implementation of an Elman network with the Heaviside non-linearity.

        DEFINITION
        A Elman language model with the Heaviside non-linearity is a tuple
        • Σ is an alphabet of symbols;
        • h0 is a D-dimensional initial vector;
        • U a D x D transition matrix;
        • V a D x |Σ| symbol matrix;
        • E a |Σ| x (D+1) emission matrix;
        • b a |Σ|-dimensional bias term.
        • f is a projection function from the unnormalized scores to the normalized
            local probability distribution over the next symbol.
        • σ is the hidden state activation function.

        Args:
            Sigma (Alphabet): The alphabet of symbols.
            U (np.ndarray): The transition matrix.
            V (np.ndarray): The symbol matrix.
            b (np.ndarray): The bias term.
            E (np.ndarray): The emission matrix.
            h0 (np.ndarray): The initial hidden state.
            f (Callable[[np.ndarray], np.ndarray]): The projection function.
            σ (Callable[[np.ndarray], np.ndarray], optional): The hidden state
                activation function. Defaults to H (Heaviside).
            one_hot (Callable[Sym, np.ndarray]): The function that maps a symbol to its
                one-hot encoding.
            n_applications (int, optional): The number of times the Elman update step
                is applied to the hidden state.
        """

        self.Sigma = Sigma

        self.U = U
        self.V = V
        self.b = b
        self.W = W
        self.E = E
        self.h0 = h0
        self.f = f
        self.sym_one_hot = one_hot
        self.σ = σ
        self.n_applications = n_applications

    # TODO: Make this work with arbitrary projection function
    def score(self, s: str) -> float:
        s = to_compatible_string(s)

        logp = 0.0  # Starting weight in a language model is 1.0
        h = self.h0

        for a in s:
            y = self.sym_one_hot(a)
            h, logits = self(h, y)
            logp_ = logits[y.argmax()] - logsumexp(logits)

            logp += logp_

        return logp

    def __call__(
        self, h: np.ndarray, y: Optional[Union[str, Sym, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if y is not None:
            if isinstance(y, str):
                y = Sym(y)

            if isinstance(y, Sym):
                y = self.sym_one_hot(y)

            for ll in range(self.n_applications):
                h = self.σ(np.dot(self.U, h) + np.dot(self.V, y) + self.b)

        hʼ = np.log(self.W @ h)
        logits = np.zeros((self.E.shape[0]))
        for j in range(self.E.shape[0]):
            _hʼ = hʼ.copy()
            _hʼ[self.E[j, :] == 0] = 0
            logits[j] = self.E[j, :] @ _hʼ

        return h, logits
