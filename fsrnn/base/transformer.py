from typing import Set

from fsrnn.base.fsa import FSA
from fsrnn.base.fst import FST
from fsrnn.base.pathsum import Pathsum, Strategy
from fsrnn.base.state import State


class Transformer:
    @staticmethod
    def _add_trim_arcs(F: FSA, T: FSA, AC: Set[State]):
        for i in AC:
            if isinstance(F, FST):
                for (a, b), j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, b, j, w)

            else:
                for a, j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, j, w)

    @staticmethod
    def trim(F: FSA) -> FSA:
        """trims the machine"""

        # compute accessible and co-accessible arcs
        A, C = F.accessible(), F.coaccessible()
        AC = A.intersection(C)

        # create a new F with only the pruned arcs
        T = F.spawn()
        Transformer._add_trim_arcs(F, T, AC)

        # add initial state
        for q, w in F.I:
            if q in AC:
                T.set_I(q, w)

        # add final state
        for q, w in F.F:
            if q in AC:
                T.set_F(q, w)

        return T

    @staticmethod
    def push(fsa):
        W = Pathsum(fsa).backward(Strategy.LEHMANN)
        pfsa = Transformer._push(fsa, W)
        # assert pfsa.pushed  # sanity check
        return pfsa

    @staticmethod
    def _push(fsa, V):
        """
        Mohri (2001)'s weight pushing algorithm. See Eqs 1, 2, 3.
        Link: www.isca-speech.org/archive_v0/archive_papers/eurospeech_2001/e01_1603.pdf
        """

        pfsa = fsa.spawn()
        for i in fsa.Q:
            pfsa.set_I(i, fsa.λ[i] * V[i])
            pfsa.set_F(i, ~V[i] * fsa.ρ[i])
            for a, j, w in fsa.arcs(i):
                if isinstance(fsa, FST):
                    pfsa.add_arc(i, a[0], a[1], j, ~V[i] * w * V[j])
                else:
                    pfsa.add_arc(i, a, j, ~V[i] * w * V[j])

        return pfsa

    @staticmethod
    def renormalize_decoupled_fst(T: FST) -> FST:
        """
        WARNING: THIS DOES NOT WORK!!!
        This was an attempt at the following:
        Given a decoupled FST, replace the weights such that, when composed with
        `string_fsa(x)`, the resulting FST represents a locally normalized FST.

        Args:
            T (FST): The decoupled FST.

        Returns:
            FST: The renormalized FST.
        """
        # TODO: (Anej) Check if the FST is decoupled.
        Tʼ = T.spawn()

        for q in T.Q:
            for (a, b), j, w in T.arcs(q):
                if q.label == "sep":
                    Tʼ.add_arc(q, a, b, j, w)
                else:
                    Tʼ.add_arc(q, a, b, j, T.R.one)

        for q, w in T.I:
            # Tʼ.add_I(q, w)
            Tʼ.add_I(q, T.R.one)
        for q, w in T.F:
            # Tʼ.add_F(q, w)
            Tʼ.add_F(q, T.R.one)

        return Tʼ
