from __future__ import annotations

import copy
from collections import Counter
from collections import defaultdict as dd
from itertools import product
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from frozendict import frozendict

import fsrnn
from fsrnn.base.pathsum import Pathsum, Strategy
from fsrnn.base.semiring import Real, Semiring
from fsrnn.base.state import State
from fsrnn.base.symbol import Sym, ε


class FSA:
    def __init__(self, R: Type[Semiring] = Real):
        # DEFINITION
        # A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
        # • R is a semiring;
        # • Σ is an alphabet of symbols;
        # • Q is a finite set of states;
        # • δ is a finite relation Q × Σ × Q × R;
        # • λ is an initial weight function;
        # • ρ is a final weight function.

        # NOTATION CONVENTIONS
        # • single states (elements of Q) are denoted q
        # • multiple states not in sequence are denoted, p, q, r, ...
        # • multiple states in sequence are denoted i, j, k, ...
        # • symbols (elements of Σ) are denoted lowercase a, b, c, ...
        # • single weights (elements of R) are denoted w
        # • multiple weights (elements of R) are denoted u, v, w, ...alphabet

        # semiring
        self.R = R

        # alphabet of symbols
        self.Sigma = set([])
        self.symbol2idx, self.idx2symbol = {}, {}

        # a finite set of states
        self.Q = set([])
        self.state2idx, self.idx2state = {}, {}

        # transition function : Q × Σ × Q → R
        self.δ = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))
        # We also define the inverse transition function δ_inv
        self.δ_inv = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        # initial weight function
        self.λ = R.chart()

        # final weight function
        self.ρ = R.chart()

    def add_state(self, q: State) -> None:
        """Adds a state to the automaton.
        This method should mainly be accessed through the add_arc method.

        Args:
            q (State): The state to be added.
        """
        assert isinstance(self.Q, set), "Cannot add to frozen FSA"
        self.Q.add(q)

    def add_states(self, Q: Union[List[State], Set[State], Tuple[State, ...]]) -> None:
        """Adds a list of states to the automaton."""
        for q in Q:
            if q not in self.state2idx:
                self.state2idx[q] = len(self.state2idx)
                self.idx2state[self.state2idx[q]] = q
            self.add_state(q)

    def add_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] += w
        self.δ_inv[j][a][i] += w

    def set_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] = w
        self.δ_inv[j][a][i] = w

    def set_I(self, q, w=None):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.λ[q] = w

    def set_F(self, q, w=None):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.ρ[q] = w

    def add_I(self, q, w):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.λ[q] += w

    def add_F(self, q, w):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.ρ[q] += w

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.Q = frozenset(self.Q)
        self.δ = frozendict(self.δ)
        self.δ_inv = frozendict(self.δ_inv)
        self.λ = frozendict(self.λ)
        self.ρ = frozendict(self.ρ)

    @property
    def I(self) -> Generator[Tuple[State, Semiring], None, None]:  # noqa: E741, E743
        """Returns the initial states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the initial states of the FSA.
        """
        for q, w in self.λ.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self) -> Generator[Tuple[State, Semiring], None, None]:
        """Returns the final states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the final states of the FSA.
        """
        for q, w in self.ρ.items():
            if w != self.R.zero:
                yield q, w

    def reverse(self):
        """creates a reversed machine"""

        # create the new machine
        R = self.spawn()

        # add the arcs in the reversed machine
        for i in self.Q:
            for a, j, w in self.arcs(i):
                R.add_arc(j, a, i, w)

        # reverse the initial and final states
        for q, w in self.I:
            R.set_F(q, w)
        for q, w in self.F:
            R.set_I(q, w)

        return R

    def arcs(
        self, i: State, no_eps: bool = False, nozero: bool = True, reverse: bool = False
    ) -> Generator[Tuple[Sym, State, Semiring], None, None]:
        """Returns the arcs stemming from state i or going into the state i in the FSA.
        in the form of tuples (a, j, w) where a is the symbol, j is the target state of
        the transition and w is the weight.

        Args:
            i (State): The state out of which the arcs stem or into which the arcs go.
            no_eps (bool, optional): If True, epsilon arcs are not returned.
                Defaults to False.
            nozero (bool, optional): If True, zero-weight arcs are not returned.
                Defaults to True.
            reverse (bool, optional): If False, the arcs stemming from state i are
                returned. If True, the arcs going into the state i are returned.
                Defaults to False.

        Yields:
            Generator[Tuple[Sym, State, Semiring], None, None]:
            Generator of the arcs stemming from state i in the FSA.
        """
        δ = self.δ if not reverse else self.δ_inv
        for a, transitions in δ[i].items():
            if no_eps and a == ε:
                continue
            for j, w in transitions.items():
                if w == self.R.zero and nozero:
                    continue
                yield a, j, w

    def predecessors(
        self, q: Union[State, Set[State]], a: Union[Sym, Set[Sym]]
    ) -> Set[State]:
        """Returns the set of predecessors of a set of states q
            for a given set of symbols a.

        Args:
            q (Union[State, Set[State]]): The set of states or an individual state
            for which the predecessors are returned.
            a (Union[Sym, Set[Sym]]): The symbols or an individual symbol for which
            the predecessors are returned.

        Returns:
            Set[State]: The set of predecessors of a state q for a given symbol a.
        """
        if not isinstance(q, set):
            q = {q}
        if not isinstance(a, set):
            a = {a}
        P = set()
        for q_, a_ in product(q, a):
            P |= set(self.δ_inv[q_][a_].keys())
        return P

    def accept(self, string: Union[str, Sequence[Sym]]) -> Semiring:
        """Determines the stringsum/acceptance weight of the string `string`
        in the rational series defined by the WFSA.

        Args:
            string (Union[str, Sequence[Sym]]):
                The string whose stringsum is to be determined.

        Returns:
            Semiring: The stringsum value.
        """
        from fsrnn.base.fsa_classes import string_fsa

        fsa_s = self.intersect(string_fsa(string, self.R))

        return Pathsum(fsa_s).pathsum(Strategy.LEHMANN)

    @property
    def num_states(self) -> int:
        """Returns the number of states of the FSA."""
        return len(self.Q)

    @property
    def num_initial_states(self) -> int:
        """Returns the number of initial states of the FSA."""
        return len(list(self.I))

    @property
    def num_final_states(self) -> int:
        """Returns the number of final states of the FSA."""
        return len(list(self.F))

    @property
    def acyclic(self):
        cyclic, _ = self.dfs()
        return not cyclic

    @property
    def deterministic(self) -> bool:
        if len(list(self.I)) != 1:
            return False
        for q in self.Q:
            counter = Counter()
            for a, _, _ in self.arcs(q):
                if a == ε:  # a deterministic fsa cannot have ε transitions
                    return False
                counter[a] += 1
            most_common = counter.most_common(1)
            if len(most_common) > 0 and most_common[0][1] > 1:
                return False
        return True

    @property
    def pushed(self) -> bool:
        for i in self.Q:
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                total += w
            if total != self.R.one:
                return False
        return True

    @property
    def probabilistic(self) -> bool:  # noqa: C901
        assert self.R == Real

        total = self.R.zero
        for i, w in self.I:
            if not w.value >= 0:
                return False, "Initial weights must be non-negative."
            total += w
        if total != self.R.one:
            return False, "Total weight of initial states must be 1."

        for i in self.Q:
            if not self.ρ[i].value >= 0:
                return False, "Final weights must be non-negative."
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                if not w.value >= 0:
                    return False, "Transition weights must be non-negative."
                total += w
            if total != self.R.one:
                return False, "Total weight of outgoing arcs must be 1."
        return True

    def copy(self):
        """deep copies the machine"""
        return copy.deepcopy(self)

    def spawn(self, keep_init=False, keep_final=False):
        """returns a new FSA in the same semiring"""
        F = FSA(R=self.R)

        if keep_init:
            for q, w in self.I:
                F.set_I(q, w)
        if keep_final:
            for q, w in self.F:
                F.set_F(q, w)

        return F

    def lift(self, R: Semiring, lifter: Callable[[Semiring], Semiring]) -> "FSA":
        """Lifts the weights of the FSA into a different different semiring where
        the weights are defined by the lifter function.

        Args:
            R (Semiring): The semiring into which the weights are lifted.
            lifter (Callable[[Semiring], Semiring]): The function that maps the original
                weights into the lifted ones.

        Returns:
            FSA: The lifted FSA.
        """
        A = FSA(R)
        for q, w in self.I:
            A.set_I(q, lifter(w))
        for q, w in self.F:
            A.set_F(q, lifter(w))
        for q in self.Q:
            for a, j, w in self.arcs(q):
                A.add_arc(q, a, j, lifter(w))
        return A

    def push(self):
        from fsrnn.base.transformer import Transformer

        return Transformer.push(self)

    def accessible(self):
        """computes the set of accessible states"""
        A = set()
        stack = [q for q, w in self.I if w != self.R.zero]
        while stack:
            i = stack.pop()
            for _, j, _ in self.arcs(i):
                if j not in A:
                    stack.append(j)
            A.add(i)

        return A

    def coaccessible(self):
        """computes the set of co-accessible states"""
        return self.reverse().accessible()

    def transition(
        self, q: State, a: Sym, weight: bool = False
    ) -> Optional[Union[State, Tuple[State, Semiring]]]:
        """If the FSA is deterministic and there exists an a-transition out of q,
            then the function returns the target state of the transition.

        Args:
            q (State): The state.
            a (Sym): The symbol.
            weight (bool, optional): Whether to return the weight of the transition.

        Returns:
            State: The target state of the transition.
        """
        assert self.deterministic

        if self.has_outgoing_arc(q, a):
            if weight:
                return list(self.δ[q][a].items())[0]
            else:
                return list(self.δ[q][a].keys())[0]
        else:
            return None

    def dfs(
        self, Is: Optional[Set[State]] = None, intervals: bool = False
    ) -> Union[
        Tuple[bool, Dict[State, int]], Tuple[bool, Dict[State, Tuple[int, int]]]
    ]:
        """Depth-first search (Cormen et al. 2019; Section 22.3)

        Args:
            Is (Optional[Set[State]], optional): The set of initial states to start
            the DFS from.
            intervals (bool, optional): Whether to return the intervals of the DFS.
            Defaults to False.

            Returns:
                Union[Tuple[bool, Dict[State, int]],
                    Tuple[bool, Dict[State, Tuple[int, int]]]]:
                If `intervals` is False, the function returns a tuple (cyclic, finished)
                where `cyclic` is a boolean indicating whether the FSA is cyclic and
                `finished` is a dictionary mapping each state to its finishing time.
                If `intervals` is True, the function returns a tuple (cyclic, finished)
                where `cyclic` is a boolean indicating whether the FSA is cyclic and
                `finished` is a dictionary mapping each state to its
                interval on the stack.
        """

        in_progress, finished = set([]), dict()
        cyclic, counter = False, 0

        def _dfs(p):
            nonlocal in_progress
            nonlocal finished
            nonlocal cyclic
            nonlocal counter

            in_progress.add(p)
            if intervals:
                finished[p] = (counter, None)

            for _, q, _ in self.arcs(p):
                if q in in_progress:
                    cyclic = True
                elif q not in finished:
                    _dfs(q)

            in_progress.remove(p)
            finished[p] = counter if not intervals else (finished[p][0], counter)
            counter += 1

        Is = Is if Is is not None else set([q for q, _ in self.I])
        for q in Is:
            _dfs(q)

        return cyclic, finished

    def finish(self, rev=False, acyclic_check=False):
        """
        Returns the nodes in order of their finishing time.
        """

        _, finished = self.dfs()

        if acyclic_check:
            assert self.acyclic

        sort = {}
        for s, n in finished.items():
            sort[n] = s
        if rev:
            for n in sorted(list(sort.keys())):
                yield sort[n]
        else:
            for n in reversed(sorted(list(sort.keys()))):
                yield sort[n]

    def toposort(self, rev=False):
        return self.finish(rev=rev, acyclic_check=True)

    def trim(self):
        from fsrnn.base.transformer import Transformer

        return Transformer.trim(self)

    def pathsum(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.pathsum(strategy)

    def backward(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.backward(strategy)

    def intersect(self, other: "FSA") -> "FSA":
        """This method performs an on-the-fly weighted intersection of two FSA.
        It works by keeping a stack of accessible states in the intersection WFSA.
        It uses the epsilon filter to handle epsilon transitions.

        Args:
            fsa (FSA): The other FSA to intersect with.

        Returns:
            FSA: The intersection FSA.
        """

        # the two machines need to be in the same semiring
        assert self.R == other.R

        return self.to_identity_fst().compose(other.to_identity_fst()).project(1)

    def to_identity_fst(self) -> "fsrnn.base.fst.FST":
        """Method converts FSA to FST.

        Returns:
            FST: FST object that accepts X:X iff `self` accepts X.
        """
        from fsrnn.base.fst import FST

        fst = FST(self.R)
        for q in self.Q:
            for a, j, w in self.arcs(q):
                fst.add_arc(i=q, a=a, b=a, j=j, w=w)
        for q, w in self.I:
            fst.set_I(q=q, w=w)
        for q, w in self.F:
            fst.set_F(q=q, w=w)
        return fst

    def __call__(self, str):
        return self.accept(str)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersect(other)

    def __or__(self, other):
        return self.union(other)

    def __repr__(self):
        return f"WFSA({self.num_states} states, {self.R})"

    def __str__(self):
        output = []
        for q, w in self.I:
            output.append(f"initial state:\t{q.idx}\t{w}")
        for q, w in self.F:
            output.append(f"final state:\t{q.idx}\t{w}")
        for p in self.Q:
            for a, q, w in self.arcs(p):
                output.append(f"{p}\t----{a}/{w}---->\t{q}")
        return "\n".join(output)

    def __getitem__(self, n):
        return list(self.Q)[n]

    def __len__(self):
        return len(self.Q)

    def _repr_html_(self):  # noqa: C901
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """
        import json
        from collections import defaultdict
        from uuid import uuid4

        from fsrnn.base.fst import FST

        def weight2str(w):
            if isinstance(w, Real):
                return f"{w.value:.3f}"
            return str(w)

        ret = []
        if self.num_states == 0:
            return "<code>Empty FST</code>"

        if self.num_states > 64:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        finals = {q for q, _ in self.F}
        initials = {q for q, _ in self.I}

        # print initial
        for q, w in self.I:
            if q in finals:
                label = f"{str(q)} / [{weight2str(w)} / {str(self.ρ[q])}]"
                color = "af8dc3"
            else:
                label = f"{str(q)} / {weight2str(w)}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{repr(q)}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )

            ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

        # print normal
        for q in (self.Q - finals) - initials:
            lbl = str(q)

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

        # print final
        for q, w in self.F:
            # already added
            if q in initials:
                continue

            if w == self.R.zero:
                continue
            lbl = f"{str(q)} / {weight2str(w)}"

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for a, j, w in self.arcs(q):
                if isinstance(self, FST):
                    label = f"{repr(a[0])}:{repr(a[1])} / {weight2str(w)}"
                else:
                    a = str(repr(a))[1:-1]
                    label = f"{a} / {weight2str(w)}"
                to[j].append(label)

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label, qrep, drep = json.dumps("\n".join(values)), repr(q), repr(d)
                ret.append(
                    f'g.setEdge("{qrep}","{drep}",{{arrowhead:"vee",label:{label}}});\n'
                )

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        ret2 = [
            """
       <script>
       try {
       require.config({
       paths: {
       "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
       "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
       }
       });
       } catch {
       ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
       "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
            function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            }
        )
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        """
        ]

        obj = "fst_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            """
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width")-g.graph().width*initialScale)/2,20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();

        </script>
        """
        )

        return "".join(ret2)
