from typing import Dict, Set, Tuple

import numpy as np
import numpy.typing as npt

from fsrnn.base.state import State
from fsrnn.base.symbol import Sym


def Q2M(
    Q: Set[State], F_2: Dict[State, Tuple[int, int]], s: int
) -> npt.NDArray[np.int_]:
    """Computes the matrix M with 1s in the entries corresponding to the pair
    encodings v of the states in Q.

    Args:
        Q (Set[State]): The set of states.
        F_2 (Dict[State, Tuple[int, int]]): The pair encoding of the states.
        s (int): The dimensionality of the pair encoding matrix.

    Returns:
        npt.NDArray[np.int_]: The matrix M.
    """
    M = np.zeros((s, s), dtype=np.int_)

    for q in Q:
        M[F_2[q]] = 1

    return M


def sym_one_hot(a: Sym, M: int, sym2idx: Dict[Sym, int]) -> npt.NDArray[np.int_]:
    y = np.zeros(M, dtype=np.int_)
    y[sym2idx[a]] = 1
    return y
