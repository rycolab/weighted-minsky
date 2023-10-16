import numpy as np
from pytest import mark

from fsrnn.base.alphabet import to_alphabet
from fsrnn.base.random import random_machine
from fsrnn.base.semiring import Boolean
from fsrnn.base.symbol import EOS
from fsrnn.constructions.dewdney import DewdneyTransform
from fsrnn.constructions.state_assignment import cover


def test_cover():
    rng = np.random.default_rng(0)

    for _ in range(50):
        M = rng.choice([0, 1], size=(500, 500), p=[0.9, 0.1])
        lines, _ = cover(M)

        assert np.sum(np.abs(np.sum(lines, axis=0) - M)) == 0


@mark.parametrize("n_states", [5, 13, 26, 50])
@mark.parametrize("alphabet_size", [2, 3, 4, 5, 6])
def test_dewdney(n_states: int, alphabet_size: int):
    for _ in range(50):
        F = random_machine(
            to_alphabet(["abcdefgh"[i] for i in range(alphabet_size)]),
            Boolean,
            n_states,
            bias=0.25,
            no_eps=True,
            trimmed=True,
            seed=33,
        )

        R = DewdneyTransform(F)
        for q in F.Q:
            h = R.q2h(q)
            for a, t, _ in F.arcs(q):
                h_ = R(h, a)
                assert (h_[: 2 * R.s] == R.q2h(t)[: 2 * R.s]).all()

                h_ = R(h_, EOS)

                assert (
                    h_[-1] == 1
                    and t in F.ρ.keys()
                    or h_[-1] == 0
                    and t not in F.ρ.keys()
                )


test_dewdney(15, 4)
