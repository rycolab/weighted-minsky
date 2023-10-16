"""Contains functions generating WFSAs of various types."""
from typing import Sequence, Type, Union

from fsrnn.base.alphabet import Alphabet
from fsrnn.base.fsa import FSA
from fsrnn.base.fst import FST
from fsrnn.base.semiring import Semiring
from fsrnn.base.state import State
from fsrnn.base.symbol import Sym, ε_1, ε_2


def string_fsa(
    y: Union[str, Sequence[Sym]], R: Type[Semiring], fst: bool = False
) -> FSA:
    """Returns a WFSA that accepts the string y.

    Args:
        y (str): The string to accept.
        R (Type[Semiring]): The semiring to use.
        fst (bool, optional): Whether to return an FST. Defaults to False.

    Returns:
        FSA: The WFSA.
    """

    A = FSA(R=R)
    for i, x in enumerate(list(y)):
        x = Sym(x) if isinstance(x, str) else (x if isinstance(x, Sym) else Sym(x._X))
        A.add_arc(State(i), x, State(i + 1), R.one)

    A.set_I(State(0), R.one)
    A.add_F(State(len(y)), R.one)

    return A if not fst else A.to_identity_fst()


def get_epsilon_filter(R: Type[Semiring], Sigma: Alphabet) -> FST:
    """Returns the epsilon filter required for the correct composition of WFSTs
    with epsilon transitions.

    Returns:
        FSA: The 3-state epsilon filter WFST.
    """

    F = FST(R)

    # 0 ->
    for a in Sigma:
        F.add_arc(State(0), a, a, State(0), R.one)
    F.add_arc(State(0), ε_2, ε_1, State(0), R.one)
    F.add_arc(State(0), ε_1, ε_1, State(1), R.one)
    F.add_arc(State(0), ε_2, ε_2, State(2), R.one)

    # 1 ->
    for a in Sigma:
        F.add_arc(State(1), a, a, State(0), R.one)
    F.add_arc(State(1), ε_1, ε_1, State(1), R.one)

    # 2 ->
    for a in Sigma:
        F.add_arc(State(2), a, a, State(0), R.one)
    F.add_arc(State(2), ε_2, ε_2, State(2), R.one)

    F.set_I(State(0), R.one)

    F.set_F(State(0), R.one)
    F.set_F(State(1), R.one)
    F.set_F(State(2), R.one)

    return F
