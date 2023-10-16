import random

import numpy as np

from fsrnn.base.alphabet import to_alphabet
from fsrnn.base.random import random_machine as rand
from fsrnn.base.semiring import Real
from fsrnn.constructions.minsky import MinskyTransform


def test_minsky_transform_transitions():
    Sigma = to_alphabet(list("ab"))

    for _ in range(10):
        # random language model
        A = rand(Sigma, Real, 32, no_eps=True, deterministic=True, trim=True)

        M = MinskyTransform(A)

        for p in A.Q:
            for b, q, _ in A.arcs(p):
                for a in A.Sigma:
                    # Encode the current state and symbol with which it was entered.
                    h = M.hidden_state(p, a)

                    h, _ = M.R(h, b)  # Get the hidden state encoding (q, b)

                    # The non-zero index of h encodes the state in which
                    # the automaton A is and the symbol with which it was entered.
                    assert (q, b) == M.n_inv[int(h.argmax())]


def test_minsky_transform_probabilities():
    Sigma = to_alphabet(list("abcd"))

    tested = 0
    while tested < 5:
        # random language model
        A = rand(
            Sigma,
            Real,
            16,
            no_eps=True,
            deterministic=True,
            trim=True,
            pushed=True,
            divide_by=10,
        )
        if len(list(A.I)) == 0:
            continue
        tested += 1

        A.λ[list(A.I)[0][0]] = A.R.one  # Set the initial state to have initial weight 1

        M = MinskyTransform(A)
        R = M.rnn()

        for _ in range(10):
            y = "".join(
                random.choice(list("abcd")) for _ in range(1, random.choice(range(6)))
            )

            Ay = A(y)
            Ry = R.score(y)

            assert abs(float(Ay) - np.exp(Ry)) < 1e-8
