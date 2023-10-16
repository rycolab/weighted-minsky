import random
from itertools import product
from math import sqrt
from typing import List, Optional, Tuple, Type, Union

from fsrnn.base.alphabet import Alphabet
from fsrnn.base.fsa import FSA, State
from fsrnn.base.fst import FST
from fsrnn.base.pathsum import Pathsum
from fsrnn.base.semiring import Boolean, Real, Semiring
from fsrnn.base.symbol import Expr, Sym, ε


def rw(semiring, **kwargs):  # noqa: C901
    if semiring is Boolean:
        return semiring(True)

    elif semiring is Real:
        tol = 1e-3
        s = kwargs.get("divide_by", 2)
        random_weight = round(random.random() / s, 3)
        while random_weight < sqrt(tol):
            random_weight = round(random.random() / s, 3)
        return semiring(random_weight)


def _add_arc(
    i: int,
    a: Union[Sym, Tuple[Sym, Sym]],
    j: int,
    used_a: List[Union[Sym, Tuple[Sym, Sym]]],
    A: Union[FSA, FST],
    bias: float = 0.25,
    acyclic: bool = False,
    deterministic: bool = True,
    **kwargs,
) -> bool:
    """Handles adding a state to the random machine.

    Args:
        i (int): _description_
        a (Sym): _description_
        j (int): _description_
        used_a (List[Sym]): _description_
        fsa (FSA): _description_
        bias (float, optional): _description_. Defaults to 0.25.
        acyclic (bool, optional): _description_. Defaults to False.
        deterministic (bool, optional): _description_. Defaults to True.
        fst (bool, optional): _description_. Defaults to False.
        kwargs: Arguments for random weight generation

    Returns:
        Whether the arc was added.
    """

    if deterministic and a in used_a:
        # always add at most one failure arc
        # or at most of any symbol if machine should be deterministic
        return False

    if random.random() < bias:
        w = rw(A.R, **kwargs)
        if acyclic:
            # Make sure that the failure arcs *always* form an acyclic subgraph
            if i < j:
                if isinstance(A, FST):
                    assert isinstance(a, tuple)
                    A.add_arc(State(i), a[0], a[1], State(j), w)
                else:
                    assert isinstance(a, Sym) or isinstance(a, Expr)
                    A.add_arc(State(i), a, State(j), w)
                used_a.append(a)
                return True
            else:
                return False
        else:
            if isinstance(A, FST):
                assert isinstance(a, tuple)
                A.add_arc(State(i), a[0], a[1], State(j), w)
            else:
                assert isinstance(a, Sym) or isinstance(a, Expr)
                A.add_arc(State(i), a, State(j), w)
            used_a.append(a)
            return True
    else:
        return False


def _add_initial_and_final(
    fsa: Union[FSA, FST], num_states: int, num_initial: int, num_final: int
):
    Is = random.choices(list(fsa.Q), k=num_initial)
    for q in Is:
        fsa.set_I(q, rw(fsa.R, divide_by=num_initial))
    Fs = random.choices(list(fsa.Q), k=num_final)
    for q in Fs:
        fsa.set_F(q, rw(fsa.R, divide_by=num_final))


def _add_pfsa_arcs(
    num_states: int,
    alphabet: Alphabet,
    i: int,
    A: FSA,
    bias: float,
    acyclic: bool,
    deterministic: bool,
) -> None:
    if deterministic:
        out_neighbourhood_size = min(
            sum([int(random.random() < bias) for _ in range(num_states)]), len(alphabet)
        )
        α = [random.random() for _ in range(out_neighbourhood_size)]
        α = [t / sum(α) for t in α]
        _alphabet = list(alphabet)
        random.shuffle(_alphabet)
        for ii, j in enumerate(
            random.sample(range(num_states), out_neighbourhood_size)
        ):
            A.add_arc(State(i), _alphabet[ii], State(j), Real(α[ii]))
    else:
        out_neighbourhood_size = sum(
            [int(random.random() < bias) for _ in range(num_states * len(alphabet))]
        )
        α = [random.random() for _ in range(out_neighbourhood_size)]
        α = [t / sum(α) for t in α]
        for ii, (j, a) in enumerate(
            random.sample(
                list(product(range(num_states), alphabet)), out_neighbourhood_size
            )
        ):
            A.add_arc(State(i), a, State(j), Real(α[ii]))


def _add_fsa_arcs(
    num_states: int,
    alphabet: Alphabet,
    i: int,
    used_a: List[Union[Sym, Tuple[Sym, Sym]]],
    A: Union[FSA, FST],
    bias: float,
    acyclic: bool,
    deterministic: bool,
    **kwargs,
) -> None:
    for j in random.choices(range(num_states), k=num_states):
        for a in alphabet:
            _add_arc(
                i=i,
                a=a,
                j=j,
                used_a=used_a,
                A=A,
                bias=bias,
                acyclic=acyclic,
                deterministic=deterministic,
                **kwargs,
            )


def _random_machine(
    Σ: Alphabet,
    R: Type[Semiring],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = False,
    acyclic: bool = False,
    deterministic: bool = True,
    num_initial: int = 1,
    num_final: int = 1,
    fst: bool = False,
    probabilistic: bool = False,
    **kwargs,
) -> Union[FSA, FST]:
    if fst:
        fsa = FST(R=R)
    else:
        fsa = FSA(R=R)

    if not no_eps:
        Σ.add(ε)
    else:
        Σ.discard(ε)

    if fst:
        alphabet = [(a, b) for a in Σ for b in kwargs.get("Delta", Σ)]
    else:
        alphabet = Σ
    # b = random.sample(Δ, 1)[0]

    for i in range(num_states):
        used_a = []
        if probabilistic:
            _add_pfsa_arcs(
                num_states=num_states,
                alphabet=alphabet,
                i=i,
                A=fsa,
                bias=bias,
                acyclic=acyclic,
                deterministic=deterministic,
            )
        else:
            _add_fsa_arcs(
                num_states=num_states,
                alphabet=alphabet,
                i=i,
                used_a=used_a,
                A=fsa,
                bias=bias,
                acyclic=acyclic,
                deterministic=deterministic,
                **kwargs,
            )

    _add_initial_and_final(fsa, num_states, num_initial, num_final)

    return fsa


def _reweigh(fsa: Union[FSA, FST]):
    Z = fsa.R.zero
    for q, w in fsa.I:
        Z += w

    for q, w in fsa.I:
        fsa.set_I(q, w / Z)

    for q, wf in fsa.F:
        Z = wf
        for a, j, w in fsa.arcs(q):
            Z += w
        for a, j, w in fsa.arcs(q):
            fsa.set_arc(q, a, j, w / Z)
        fsa.set_F(q, wf / Z)


def random_pfsa(
    Sigma: Union[Alphabet, str],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = False,
    acyclic: bool = False,
    deterministic: bool = True,
    num_initial: int = 1,
    num_final: int = 1,
    seed: Optional[int] = None,
    **kwargs,
) -> FSA:
    return random_machine(
        Sigma=Sigma,
        R=Real,
        num_states=num_states,
        bias=bias,
        no_eps=no_eps,
        acyclic=acyclic,
        deterministic=deterministic,
        probabilistic=True,
        num_initial=num_initial,
        num_final=num_final,
        seed=seed,
        **kwargs,
    )


def random_machine(
    Sigma: Union[Alphabet, str],
    R: Type[Semiring],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = False,
    eigen: bool = False,
    acyclic: bool = False,
    deterministic: bool = True,
    trimmed: bool = True,
    pushed: bool = False,
    num_initial: int = 1,
    num_final: int = 1,
    probabilistic: bool = False,
    fst: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[FSA, FST]:
    """
    Creates a random WFSA or WFST.
    It takes a number of parameters that control the properties of the machine.

    Args:
        Sigma (Alphabet): The alphabet of the WFSA.
        R (Type[Semiring]): The semiring of the WFSA.
        num_states (int): The number of states of the WFSA.
        bias (float, optional): The probability of realising an edge between
            any pair of states (q, q') with a specific symbol. Defaults to 0.25.
        no_eps (bool, optional): When true, the WFSA contains no ε transitions.
            Defaults to False.
        eigen (bool, optional): _description_. Defaults to False.
        acyclic (bool, optional): When true, the WFSA will be acyclic by design.
            Defaults to False.
        deterministic (bool, optional): When true, the WFSA will be deterministic.
            Defaults to True.
        trimmed (bool, optional): When true, trims the machine to make it smaller.
            Defaults to True.
        pushed (bool, optional): When true, pushes the machine to make it locally
            normalised. Defaults to False.
        num_initial: The number of initial states. Each will be assigned a random
            initial weight. Defaults to 1.
        num_final: The number of final states. Each will be assigned a random
            final weight. Defaults to 1.
        probabilistic: Whether to generate a probabilistic WFSA. Requires the real
            semiring. If true, the WFSA will be pushed and the initial states will
            be reweighted such that the initial weights form a probability distribution.
        fst (bool, optional): Whether to create a random _transducer_.
            Defaults to False.
        seed (int, optional): The seed for the random number generator.
        kwargs: Arguments for random weight generation

    Returns:
        Union[FSA, FST]: A random WFSA of WFST.
    """
    assert R is Real or not probabilistic

    if isinstance(Sigma, str):
        Sigma = Alphabet(Sigma)

    random.seed(seed)

    fsa = None
    while fsa is None or not fsa.num_states:
        fsa = _random_machine(
            Sigma,
            R,
            num_states,
            bias=bias,
            no_eps=no_eps,
            acyclic=acyclic,
            deterministic=deterministic,
            num_initial=num_initial,
            num_final=num_final,
            fst=fst,
            probabilistic=probabilistic,
            **kwargs,
        )

        # Trim the machine to make it smaller
        if trimmed:
            fsa = fsa.trim()

        if eigen and R is Real:
            pathsum = Pathsum(fsa)
            if pathsum.max_eval() >= 1.0:
                fsa = None

    if pushed:
        fsa = fsa.push()

    if probabilistic:
        _reweigh(fsa)

    return fsa
