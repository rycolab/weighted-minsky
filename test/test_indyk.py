import numpy as np
from pytest import mark

from fsrnn.base.alphabet import to_alphabet
from fsrnn.base.random import random_machine
from fsrnn.base.semiring import Boolean
from fsrnn.base.symbol import EOS
from fsrnn.constructions.indyk import IndykTransform

rng = np.random.default_rng(0)


@mark.parametrize("n_states", [30, 50, 100, 200])
@mark.parametrize("alphabet_size", [2, 3, 4, 5, 6])
def test_indyk(n_states: int, alphabet_size: int):
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

        R = IndykTransform(F, rng)

        for q in F.Q:
            h = R.q2h(q)
            for a, t, _ in F.arcs(q):
                h_ = R(h, a)
                assert (h_[: 4 * R.r] == R.q2h(t)[: 4 * R.r]).all()

                h_ = R(h_, EOS)

                assert (
                    h_[-1] == 1
                    and t in F.ρ.keys()
                    or h_[-1] == 0
                    and t not in F.ρ.keys()
                )


test_indyk(30, 6)
