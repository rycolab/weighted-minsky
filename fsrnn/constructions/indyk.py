# Construction of an RNN representing an arbitrary deterministic finite automaton
# with N states using O(sqrt(N)) cells in the hidden state.
from itertools import product
from math import ceil
from typing import List, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from fsrnn.base.fsa import FSA
from fsrnn.base.state import State
from fsrnn.base.symbol import EOS, Sym
from fsrnn.constructions import utils as fsa_utils
from fsrnn.models.activations import H

RandomGenerator = Union[np.random.Generator, int]


class IndykTransform:
    def __init__(self, A: FSA, rng: RandomGenerator) -> None:
        assert EOS not in A.Sigma

        self.rng = np.random.default_rng(rng)

        self.A = A
        self.Σ = list(self.A.Sigma.union({EOS}))
        self.M = len(self.Σ)  # The size of the alphabet with the EOS symbol
        self.sym2idx = {sym: idx for idx, sym in enumerate(self.Σ)}

        self.r = ceil(self.A.num_states ** (1 / 4))
        self.r2 = self.r**2

        self.D_u = 4 * self.r + 1  # +1 for the acceptance bit
        self.D_v = 2 * self.r2

        # The permutation for defining the F4 tuple function in the Indyk transform
        # self.pi = rng.permutation(self.A.num_states)
        self.pi = np.arange(self.A.num_states)
        self.pi_inv = np.argsort(self.pi)

        self.build_F()

        self.build_pair_representation_matrix()

        # Create the matrix simulating the automaton's transition function
        self.build_recurrence_parameters()

        # Create the input symbol matrix
        self.build_input_matrix()

        # Create the matrix deciding the acceptance of the input string
        self.build_acceptance_matrix()

        self.h0 = np.zeros(self.D_u, dtype=np.int_)
        self.build_initial_state()

    def build_initial_state(self):
        """Builds the initial state of the RNN."""
        q0 = [q for q, _ in self.A.I][0]
        self.h0 = self.q2h(q0)

    def q2h(self, q: State) -> npt.NDArray[np.int_]:
        """Converts a state in the FSA to a hidden state."""
        h = np.zeros(self.D_u, dtype=np.int_)
        # The acceptance bit is initially set to 0 and is only set to one after reading
        # in the EOS symbol.
        h[:-1] = self.q2u(q)
        return h

    def h2q(self, h: npt.NDArray[np.int_]) -> State:
        """Returns the state corresponding to the hidden state h.

        Args:
            h (npt.NDArray[np.int_]): The hidden state.

        Returns:
            State: The state corresponding to the hidden state h.
        """
        for j in range(4):
            assert all(
                np.unique(h[j * self.r : j * self.r + self.r]) == np.array([0, 1])
            )
            assert np.sum(h[j * self.r : j * self.r + self.r]) == 1

        r = tuple(h[j * self.r : j * self.r + self.r].argmax() for j in range(4))
        assert r in self.F_4_inv

        return self.F_4_inv[r]

    def build_F(self) -> None:
        """Builds the functions F_4 mapping states to 4-tuples of integers in [0, r) and
        F_2, mapping states to pairs of integer in [0, r^2)."""
        self.F_4, self.F_4_inv = dict(), dict()
        self.F_2, self.F_2_inv = dict(), dict()

        for q in self.A.Q:
            (l1, l2, l3, l4) = (
                self.pi[self.A.state2idx[q]] % self.r,
                self.pi[self.A.state2idx[q]] // self.r % self.r,
                self.pi[self.A.state2idx[q]] // self.r**2 % self.r,
                self.pi[self.A.state2idx[q]] // self.r**3 % self.r,
            )

            self.F_4[q] = (l1, l2, l3, l4)
            self.F_4_inv[self.F_4[q]] = q

            self.F_2[q] = (self.r * l1 + l2, self.r * l3 + l4)
            self.F_2_inv[self.F_2[q]] = q

    def q2u(self, q: State) -> npt.NDArray[np.int_]:
        """Constructs the 4r-dimensional vector u(q) as defined in the Indyk transform,
        i.e., the vector with entries u^j_l with u^j_l = 1 if F4(q)_j = l
        and 0 otherwise.

        Args:
            q (State): The state q.

        Returns:
            npt.NDArray[np.int_]: The vector u(q).
        """
        u = np.zeros(4 * self.r, dtype=np.int_)
        (l1, l2, l3, l4) = self.F_4[q]
        u[[l1, self.r + l2, 2 * self.r + l3, 3 * self.r + l4]] = 1
        return u

    def q2v(self, q: State) -> npt.NDArray[np.int_]:
        """Constructs the 2r^2-dimensional vector v(q) as defined in the
        Indyk transform, i.e., the vector with entries v^j_k with
        v^j_{l = r l + l'} = u^{2j-1}_{l} * u^{2j}_{l'}.
        Given q, this vector has 1 on coordinates corresponding to r*l_1 + l_2 and
        r*l_3 + l_4, where (l_1, l_2, l_3, l_4) = F4(q).
        It is analogous to the vector u(q) but with the coordinates computed by F2.

        Args:
            q (State): The state q.

        Returns:
            npt.NDArray[np.int_]: The vector v(q).
        """
        v = np.zeros(2 * self.r**2, dtype=np.int_)
        (l1, l2, l3, l4) = self.F_4[q]
        v[self.r * l1 + l2] = 1
        v[self.r**2 + self.r * l3 + l4] = 1
        return v

    def build_pair_representation_matrix(self):  # TODO: this does not work
        """
        Builds the representation matrix S based on the two-dimensional representations
        of the states v.
        """
        self.S = -1 * np.ones((self.r2, self.r2), dtype=np.int_)
        for q in self.A.state2idx:
            if isinstance(q.idx, int):
                self.S[self.F_2[q]] = q.idx

    def display_state_encoding(self):
        """Displays the state encoding matrix."""
        print(self.S)

    def display_cell_state(self, h: npt.NDArray[np.int_]):
        """Displays the cell state (one dimension for each f_i).

        Args:
            h (npt.NDArray[np.int_]): The hidden state containing the activations
            of individual f_i.
        """
        print()
        c = 0
        for j, a, l in product(range(4), self.Σ, range(self.r)):
            for jj in range(self.fs_per_cell[(j, a, l)]):
                states_of_interest = self.projection_to_states(j, a, l)
                print(f"h({jj} | {states_of_interest} ({j}, {a}, {l})) = {h[c + jj]}")
            c += self.fs_per_cell[(j, a, l)]
        print()

    def display_candidates_state(self, h: npt.NDArray[np.int_]):
        """Displays the candidates state (one dimension for each possible (j, a, l)).

        Args:
            h (npt.NDArray[np.int_]): The hidden state containing the activations
            of individual cells corresponding to (j, a, l).
        """
        print()
        for ii, (j, a, l) in enumerate(product(range(4), self.Σ, range(self.r))):
            states_of_interest = self.projection_to_states(j, a, l)
            print(f"h({states_of_interest} ({j}, {a}, {l})) = {h[ii]}")
        print()

    def projection_to_states(self, j: int, a: Sym, l_: int) -> Set[State]:
        """Constructs the set of states whose 4-tuple encoding has the j-th coordinate
        equal to l.

        Args:
            j (int): The coordinate in the 4-tuple encoding.
            a (int): The label on the transitions of interest.
            l (int): The value of the j-th coordinate of interest.

        Returns:
            Set[State]: The set of states whose 4-tuple encoding has the j-th coordinate
            equal to l.
        """
        states_of_interest = set()

        # Loop through all possible 4-tuples with the j-th coordinate equal to l.
        for l1, l2, l3 in product(range(self.r), repeat=3):
            # Set the j-th coordinate to l and the rest to l1, l2, l3
            t = np.asarray((-1, -1, -1, -1))
            t[j] = l_
            t[t == -1] = [l1, l2, l3]
            t = tuple(t)

            # Get the state corresponding to the 4-tuple
            if t in self.F_4_inv:
                states_of_interest.add(self.F_4_inv[t])

        return states_of_interest

    def predecessors(self, j: int, a: Sym, l_: int) -> Set[State]:
        """Constructs the set A_{jla} as defined in the Indyk transform, i.e.,
        the set of states from which we can reach, with the symbol a,
         a state whose 4-tuple encoding has the j-th coordinate equal to l.

        Args:
            j (int): The coordinate in the 4-tuple encoding.
            a (int): The label on the transitions of interest.
            l (int): The value of the j-th coordinate of interest.

        Returns:
            Set[State]: The set A_{jla}.
        """

        states_of_interest = self.projection_to_states(j, a, l_)

        # Return the predecessors of the states in states_of_interest if the symbol
        # is a normal symbol from the FSA's alphabet.
        # Otherwise, just return the original states states_of_interest, since we assume
        # that the EOS symbol does not trigger a transition in the FSA, but simply
        # signals the end of the input.
        return (
            self.A.predecessors(states_of_interest, a)
            if a != EOS
            else states_of_interest
        )

    def build_input_matrix(self):
        """Builds the input symbol matrix."""
        # Copies the one-hot encoding of the input symbols to the last part of the
        # hidden state
        self.V = -1 * np.ones((4 * self.M * self.r, self.M), dtype=np.int_)
        for j, a in product(range(4), self.Σ):
            ix = self.sym2idx[a]
            offset = j * self.M * self.r + ix * self.r
            self.V[offset : offset + self.r, ix] = 0

    def build_u2v_parameters(self):
        """Creates the matrix which transforms the vector u(q) to the vector v(q)."""
        self.U_uv = np.zeros((2 * self.r2, self.D_u), dtype=np.int_)
        for q in self.A.state2idx:
            u = self.q2u(q)
            p = self.F_2[q]
            # This ensures that the rows of the matrix corresponding to the encoding
            # of v pick up the relevant part of the encoding u (the first half for the
            # first coordinate of v and the second half for the second coordinate of v)
            self.U_uv[p[0], np.where(u[: 2 * self.r] == 1)[0]] = 1
            self.U_uv[
                self.r2 + p[1], 2 * self.r + np.where(u[2 * self.r :] == 1)[0]
            ] = 1

        # Since both coordinates of the part of u relevant to the encoding of v
        # must be activated, set the bias to -1
        self.b_uv = -1 * np.ones((2 * self.r2,), dtype=np.int_)

    def build_f_detector(self, f_matrices: List[npt.NDArray[np.int_]]):
        """This function constructs the parameters of the three layers that detect
        the activation of a single function f_i for a specific tuple (j, a, l).
        Since we have to test equality Ux = b, we have to implement this
        as a three-step process:
        1. Detect y_1 := Ux >= b <=> Ux - b >= 0
        2. Detect y_2 := -Ub >= -b <=> -Ux + b >= 0
        3. Detect Ux = b <=> y_1 and y_2

        Here, b = self.r2 + 1, and U is the matrix whose rows are formed from the
        individual f_i function detectors (see build_f).
        """

        self.U_f_1 = np.vstack(f_matrices)
        # By construction, the sum of the two factors must be at least self.r2 + 1,
        # so the bias is set to -self.r2
        self.b_f_1 = -self.r2 * np.ones((self.U_f_1.shape[0],), dtype=np.int_)

        self.U_f_2 = -1 * np.vstack(f_matrices)
        # If the sum of the two factors is exactly self.r2 + 1, then
        # self.U_f_2 x = self.r2 + 1, and we have to add + 1 to make it go above 0
        self.b_f_2 = (self.r2 + 2) * np.ones((self.U_f_2.shape[0],), dtype=np.int_)

        # We are conjugating the two halves, both values have to be 1, thus bias is -1
        self.b_f = -np.ones((self.U_f_1.shape[0],), dtype=np.int_)

    def build_candidates_matrix(
        self,
        cs_k: List[int],
    ):
        """Constructs the matrix which ORs the activations of the entries corresponding
        to individual f_i of the 4r sets A_{jla}.
        A cell refers to one of the 4r entries of the 4-tuple state encoding.
        With this, the encodings of the *candidates* for the next states are activated;
        these candidates are then filtered based on the input symbol.
        """

        assert len(cs_k) == 4 * self.M * self.r
        self.U_candidates = np.zeros((4 * self.M * self.r, sum(cs_k)), dtype=np.int_)
        for ii, (c_k, c_f) in enumerate(zip(cs_k, np.cumsum(cs_k))):
            self.U_candidates[ii, c_f - c_k : c_f] = 1

        self.b_candidates = np.zeros((4 * self.M * self.r,), dtype=np.int_)

    def build_recurrence_parameters(self):
        self.build_u2v_parameters()

        f_matrices = self.detect_f_activations()

        self.build_f_detector(f_matrices)

        self.build_candidates_matrix([f_matrix.shape[0] for f_matrix in f_matrices])

        self.build_next_state_detection()

    def build_next_state_detection(self):
        self.U_state = np.zeros((4 * self.r, 4 * self.M * self.r), dtype=np.int_)

        for j, a, l in product(range(4), self.Σ, range(self.r)):
            ix = self.sym2idx[a]
            self.U_state[j * self.r + l, j * self.M * self.r + ix * self.r + l] = 1

        # Since at most one symbol subvector can be active, we set the bias to 0
        self.b_state = np.zeros((4 * self.r,), dtype=np.int_)

        return self.U_state, self.b_state

    def detect_f_activations(self) -> List[npt.NDArray[np.int_]]:
        fs_matrices = []

        self.fs_per_cell = dict()

        # The pattern should always be [0, 1, 2, 3] x Σ x [0, ..., r-1]
        for j, a, l in product(range(4), self.Σ, range(self.r)):
            fs_matrices.append(self.build_fs(j, a, l))

        return fs_matrices

    def build_acceptance_matrix(self):
        """Builds the matrix deciding the acceptance of the input string, that is,
        it checks if any of the final states is active.
        """

        # TODO: add acceptance based on reading in the EOS symbol

        self.U_accept = np.zeros((1, 4 * self.r), dtype=np.int_)
        for q, _ in self.A.F:
            self.U_accept[0, self.q2u(q) == 1] = 1

        # Since all four coordinates of the 4-tuple encoding of the final states
        # must be activated, set the bias to -3 for the EOS symbol and -4 for all other
        # symbols, since by the definition of the RNN language model, a string can only
        # be accepted after reading in the EOS symbol.
        self.V_accept = np.zeros((1, self.M), dtype=np.int_)
        for a in self.Σ:
            if a == EOS:
                # This still makes acceptance possible if all four coordinates of the
                # 4-tuple encoding correspond to a final state.
                self.V_accept[0, self.sym2idx[a]] = -3
            else:
                # This makes acceptance impossible for non-EOS symbols.
                self.V_accept[0, self.sym2idx[a]] = -4

    def compute_interval(self, f: npt.NDArray[np.int_], k: int) -> int:
        interval_right_boundaries = np.where(np.diff(f) > 0)[0]
        interval_right_boundaries = np.append(interval_right_boundaries, self.r2 - 1)

        if f[k] == -1:
            return 0

        interval = 0 if f[0] == -1 else 1  # the first interval is 1.
        # It is only 0 if some values of k do not have assigned values f[k].
        for i in interval_right_boundaries:
            if f[k] > f[i]:
                interval += 1
            else:
                break
        return interval

    def build_f(
        self, B: npt.NDArray[np.int_]
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        # This puts a -1 in the entries corresponding to the columns (values k) with
        # no ones.
        f = B.argmax(axis=0) - (B.max(axis=0) == 0)

        # Remove the ones captured in f from B:
        rows = f[f != -1]
        cols = np.where(f != -1)[0]
        B[rows, cols] = 0

        args = f.argsort()
        P = np.eye(self.r2, dtype=np.int_)[args]

        f_sorted = P @ f

        β = np.asarray([self.compute_interval(f_sorted, k) for k in range(self.r2)])
        α = np.zeros((self.r2,), dtype=np.int_)
        for k in range(self.r2):
            if f_sorted[k] != -1:  # Only assign values to the k's that have a value.
                α[f_sorted[k]] = self.r2 + 1 - β[k]

        β = P.T @ β

        w = np.hstack((α, β))

        return w, B

    def build_fs(self, j: int, a: Sym, l_: int) -> npt.NDArray[np.int_]:
        A = self.predecessors(j, a, l_)
        B = fsa_utils.Q2M(A, self.F_2, self.r2)

        self.fs_per_cell[(j, a, l_)] = 0

        # This also works if the set states of interest (in the row/column k) is empty
        if np.sum(B) == 0:
            # There are no predecessors of the states in this row/column.
            return np.zeros((0, 2 * self.r2), dtype=np.int_)

        Us_a = []

        while np.sum(B) > 0:  # This will repeat exactly c times, where c is the
            # quantity of defined in Indyk (1995).
            w, B = self.build_f(B)
            assert np.min(B) >= 0
            Us_a.append(w)

        self.fs_per_cell[(j, a, l_)] = len(Us_a)

        U_a = np.vstack(Us_a)

        return U_a

    def __call__(self, h: npt.NDArray[np.int_], a: Sym) -> npt.NDArray[np.int_]:
        """Computes the next hidden state given the current hidden state and the
        input symbol.

        Args:
            h (npt.NDArray[np.int_]): The current hidden state.
            a (Sym): The input symbol.

        Returns:
            npt.NDArray[np.int_]: The next hidden state.
        """

        # Phase 1: Convert the ternary state encoding into a pair encoding
        h_v = H(self.U_uv @ h + self.b_uv)

        # Phase 2: Compute the activations of the individual f_i by testing the
        # equality from the Indyk paper (bottom of page 343)
        h_f_1 = H(self.U_f_1 @ h_v + self.b_f_1)
        h_f_2 = H(self.U_f_2 @ h_v + self.b_f_2)
        h_f = H(h_f_1 + h_f_2 + self.b_f)

        # Phase 3: Compute the activations of the all candidate (j, a, l) tuples,
        # where (j, a, l) encodes that the states with the value l in the j-th
        # dimension of the ternary encoding can be reached by reading in the symbol a.
        h_candidates = H(self.U_candidates @ h_f + self.b_candidates)

        # Phase 4: Add the input symbol representation and keep only the activations
        # of the tuples (j, a, l) with the same symbol.
        h_states_symbols = H(
            h_candidates + self.V @ fsa_utils.sym_one_hot(a, self.M, self.sym2idx)
        )

        # Phase 5: Convert the activations of the tuples (j, a, l) into the activations
        # of the ternary encodings of the states.
        h_states = H(self.U_state @ h_states_symbols + self.b_state)

        # Phase 6: Compute the acceptance of the input string.
        accept = H(
            self.U_accept @ h_states
            + self.V_accept @ fsa_utils.sym_one_hot(a, self.M, self.sym2idx)
        )

        h = np.hstack((h_states, accept))

        return h
