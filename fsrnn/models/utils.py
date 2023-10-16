from typing import List

from fsrnn.base.symbol import EOS, Sym, to_sym


def to_compatible_string(s: str) -> List[Sym]:
    """Converts a string to a list of symbols and appends EOS.
    Due to the way the Elman network is defined, the input string
    must not contain the BOS symbol - it is assumed that this symbol has been
    read when reading the initial hidden state h0.
    Args:
        s (str): The input string

    Returns:
        List[Sym]: List of symbols.
    """
    return [to_sym(a) for a in s] + [EOS]
