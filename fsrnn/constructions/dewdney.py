from itertools import product
from math import ceil, sqrt
from typing import List, Set

import numpy as np
import numpy.typing as npt

from fsrnn.base.fsa import FSA
from fsrnn.base.semiring import Boolean
from fsrnn.base.state import State
from fsrnn.base.symbol import EOS, Sym
from fsrnn.constructions import state_assignment as sa
from fsrnn.constructions import utils as fsa_utils
from fsrnn.models.activations import H

# TODO (Anej): Base acceptance of the string on reading the EOS symbol.


class DewdneyTransform:
    def __init__(self, A: FSA):
        assert A.R == Boolean
        assert A.deterministic

        self.A = A
        self.Σ = list(self.A.Sigma.union({EOS}))
        self.M = len(self.Σ)
        self.sym2idx = {sym: idx for idx, sym in enumerate(self.Σ)}

        self.s = ceil(sqrt(len(A.Q)))

        # The hidden state of the RNN has 2 * s + 1 neurons.
        # The first s neurons encode the activations of the rows of the FSA state
        # representation matrix, the next s neurons encode the activations of the
        # columns of the FSA state representation matrix, and the last neuron encodes
        # the acceptance of the string read so far.
        self.D = 2 * self.s + 1

        # The mapping from the FSA states to the two-dimensional state representations
        # used to identify the cells in the RNN hidden state.
        # F: Q -> {0, 1}^{s x s}
        self.F_2, self.F_2_inv = dict(), dict()
        # We also construct a matrix whose cells contain the
        # state indices
        self.S = -1 * np.ones((self.s, self.s), dtype=int)
        self.build_state_encoding()

        # Create the matrix simulating the automaton's transition function
        self.build_recurrence_matrices()

        # Create the input symbol matrix
        self.build_input_matrix()

        # Create the matrix deciding the acceptance of the input string
        self.build_acceptance_matrix()

        self.h0 = np.zeros(self.D, dtype=int)
        self.build_initial_state()

    def build_initial_state(self):
        """Builds the initial state of the RNN."""
        q0 = [q for q, _ in self.A.I][0]
        self.h0 = self.q2h(q0)

    def q2h(self, q: State) -> npt.NDArray[np.int_]:
        """Converts a state in the FSA to a hidden state."""
        e = self.F_2[q]
        h = np.zeros(self.D, dtype=int)
        h[e[0]] = h[self.s + e[1]] = 1
        return h

    def build_input_matrix(self):
        """Builds the input symbol matrix."""
        # Copies the one-hot encoding of the input symbols to the last part of the
        # hidden state
        self.V = -1 * np.ones((2 * self.M * self.s, self.M), dtype=np.int_)
        for j, a in product(range(2), self.Σ):
            ix = self.sym2idx[a]
            offset = j * self.M * self.s + ix * self.s
            self.V[offset : offset + self.s, ix] = 0

    def encoding_empty(self, k: int, row: bool = True) -> bool:
        """Checks whether the row or column k of the state encoding matrix is empty.
        This means that there will be no states whose predecessors are of interest.

        Args:
            k (int): The state number.
            row (bool, optional): Whether to check the row or the column.
            Defaults to True.

        Returns:
            bool: Whether the row or column is empty.
        """
        if row:
            return np.max(self.S[k, :]) == -1
        else:
            return np.max(self.S[:, k]) == -1

    def build_recurrence_matrices(self):
        """Builds the part of the RNN that simulates the transition function
        of the input automaton. That is, it constructs the 2 * s matrices,
        each one detecting the activation of one row or column of the encoding matrix
        (equivalently, one of the cells of the hidden state cells encoding the
        state neurons).
        """

        U1s, U2s = [], []
        for row, a, k in product([True, False], self.Σ, range(self.s)):
            predecessors = self.predecessors(k, a, row)
            B = fsa_utils.Q2M(predecessors, self.F_2, self.s)
            U_1, U_2 = sa.build_state_set_detector(B)
            U1s.append(U_1)
            U2s.append(U_2)

        # This matrix detect the activation of individual factors of the transversals
        self.U_factors = np.concatenate(U1s, axis=0)
        # The sum of the two coordinates must be at least self.s + 1
        self.b_factors = -self.s * np.ones((self.U_factors.shape[0]), dtype=int)
        # U_factors.shape = (# total transversals across M and Σ, self.D)

        transversals_per_coordinate = [U_1.shape[0] // 2 for U_1 in U1s]
        total_n_transversals = sum(transversals_per_coordinate)
        # This matrix converts the activations of the transversal factors into
        # the activations of the transversals they represent.
        self.U_transversals = np.zeros(
            (total_n_transversals, 2 * total_n_transversals), dtype=int
        )
        # This creates a matrix which conjuncts subsequent 2 elements in the vector
        # with the stride 2.
        # It has the form:
        # [[1, 1, 0, 0, ...],
        #  [0, 0, 1, 1, ...],
        #  [0, 0, 0, 0, ...],
        self.U_transversals[:, range(0, 2 * total_n_transversals, 2)] = np.eye(
            total_n_transversals
        )
        self.U_transversals[:, range(1, 2 * total_n_transversals, 2)] = np.eye(
            total_n_transversals
        )
        # Both factors must be active, so the this ensures the transversal is active
        # if and only if both factors are active.
        self.b_transversals = -1 * np.ones((total_n_transversals,), dtype=int)
        # U_transversals.shape =
        # (# total transversals across M and Σ, 2 * # total transversals across M and Σ)

        # This matrix detects the activation of the non-transversal lines,
        # i.e. the row and column matrices.`
        non_transversals_per_coordinate = [U_2.shape[0] for U_2 in U2s]
        self.U_non_transversals = np.concatenate(U2s, axis=0)
        # The sum of the two coordinates must be at least self.s + 1
        self.b_non_transversals = -self.s * np.ones(
            (self.U_non_transversals.shape[0]), dtype=int
        )
        # U_non_transversals.shape =
        # (# total non-transversal lines across M and Σ, self.D)

        # This matrix detects the activation of the cells in the hidden state
        # corresponding to the encodings of the states of the automaton,
        # with one cell for each possible input symbol.
        total_n_lines = self.U_transversals.shape[0] + self.U_non_transversals.shape[0]
        # For each cell, the activation of any of its lines is sufficient,
        # so the U_cells matrix must be sensitive to all of them.
        # However, we again have construct it in a similar striding fashion
        # as the U_transversals matrix.
        self.U_cells = np.zeros((self.M * 2 * self.s, total_n_lines), dtype=int)
        # The vector H(self.U_cells @ h_prime + self.b_cells) will contain *two*
        # entries for each row/column (entry in the hidden state): one for each possible
        # input symbol.
        # Those two entries then have to be combined to get the final activation
        # of the state neuron based on the input symbol.
        self.build_cell_matrix(
            transversals_per_coordinate, non_transversals_per_coordinate
        )
        # At least one line must be active, so we set the bias to 0
        self.b_cells = np.zeros((self.M * 2 * self.s,), dtype=int)
        # U_cells.shape =
        # ((|Σ| + 1) * 2 * self.s + self.S + 1,
        # # total transversals + # total non-transversal lines),
        # with the last self.S + 1 dimensions corresponding to the
        # one-hot encoding of the input symbol.

        self.U_states, self.b_states = sa.detect_next_state(
            self.Σ, self.s, self.M, self.sym2idx
        )

    def build_cell_matrix(
        self,
        transversals_per_coordinate: List[int],
        non_transversals_per_coordinate: List[int],
    ):
        assert len(transversals_per_coordinate) == len(non_transversals_per_coordinate)
        total_n_transversals = sum(transversals_per_coordinate)
        transversals_c, non_transversals_c = 0, 0
        for ii, (n_transversals, n_non_transversals) in enumerate(
            zip(transversals_per_coordinate, non_transversals_per_coordinate)
        ):
            # This puts 1's in the cells which will detect the activation of the
            # transversal and non-transversal lines of the ii-th coordinate
            # (row or column of the state encoding matrix).
            # Recall that the vector transformer by this matrix if of the form:
            # [# total transversal lines across M, # total non-transversal lines]

            self.U_cells[ii, transversals_c : transversals_c + n_transversals] = 1
            self.U_cells[
                ii,
                total_n_transversals
                + non_transversals_c : total_n_transversals
                + non_transversals_c
                + n_non_transversals,
            ] = 1
            transversals_c += n_transversals
            non_transversals_c += n_non_transversals

    def build_acceptance_matrix(self):
        """Builds the matrix deciding the acceptance of the input string, that is,
        it checks if any of the final states is active.
        """

        # TODO: add acceptance based on reading in the EOS symbol

        self.U_accept = np.zeros((1, 2 * self.s), dtype=int)
        final = [q for q, _ in self.A.F]
        for i, j in product(range(self.s), range(self.s)):
            # Detects whether the encoding of a final state is activated
            if (i, j) in self.F_2_inv and self.F_2_inv[(i, j)] in final:
                self.U_accept[0, i] = 1
                self.U_accept[0, self.s + j] = 1

        # Since both coordinates of the pair encoding of the final states
        # must be activated, set the bias to -1 for the EOS symbol and -2 for all other
        # symbols, since by the definition of the RNN language model, a string can only
        # be accepted after reading in the EOS symbol.
        self.V_accept = np.zeros((1, self.M), dtype=np.int_)
        for a in self.Σ:
            if a == EOS:
                # This still makes acceptance possible if all four coordinates of the
                # pair encoding correspond to a final state.
                self.V_accept[0, self.sym2idx[a]] = -1
            else:
                # This makes acceptance impossible for non-EOS symbols.
                self.V_accept[0, self.sym2idx[a]] = -2

    def build_state_encoding(self):
        """
        Builds the state encoding, i.e., the mapping from states to the two-dimensional
        representations (i, j) with i, j in {0, ..., sqrt(Q) - 1}.
        """
        for q, i in self.A.state2idx.items():
            self.F_2[q] = (i // self.s, i % self.s)
            self.F_2_inv[(i // self.s, i % self.s)] = q
            if isinstance(q.idx, int):
                self.S[i // self.s, i % self.s] = q.idx

    def display_state_encoding(self):
        """Displays the state encoding matrix."""
        print(self.S)

    def display_cell_h(self, h: npt.NDArray[np.int_]):
        """Displays and labels the cells in the
        hidden state corresponding to Phase 4.2."""
        print()
        for ii, (row, a, k) in enumerate(product([True, False], self.Σ, range(self.s))):
            print(f"Cell ({str(row):<5}, {str(a):<3}, {k}): {h[ii]}")
        print()

    def display_state_alphabet_h(self, h: npt.NDArray[np.int_]):
        """Displays and labels the cells in the
        hidden state corresponding to Phase 4.4."""
        print()
        for ii, (row, a, k) in enumerate(product([True, False], self.Σ, range(self.s))):
            print(f"Cell ({str(row):<5}, {str(a):<3}, {k}): {h[ii]}")
        print()

    def h2q(self, h: npt.NDArray[np.int_]) -> State:
        """Returns the state corresponding to the hidden state h.

        Args:
            h (npt.NDArray[np.int_]): The hidden state.

        Returns:
            State: The state corresponding to the hidden state h.
        """

        assert all(np.unique(h[: self.s]) == np.array([0, 1]))
        assert all(np.unique(h[self.s : 2 * self.s]) == np.array([0, 1]))
        assert np.sum(h[: self.s]) == 1
        assert np.sum(h[self.s : 2 * self.s]) == 1

        r = (h[: self.s].argmax(), h[self.s : 2 * self.s].argmax())
        assert r in self.F_2_inv

        return self.F_2_inv[r]

    @property
    def encoding_matrix(self) -> npt.NDArray[np.int_]:
        """Returns the encoding matrix of the states, i.e., the matrix with entries
        M_{i, j} = q if (i, j) is the encoding of q, and -1 otherwise.

        Returns:
            npt.NDArray[np.int_]: The encoding matrix of the states.
        """
        M = np.zeros((self.s, self.s), dtype=int) - 1
        for q, (i, j) in self.F_2.items():
            M[i, j] = str(q)

        return M

    def projection_to_states(self, row: bool, x: int) -> Set[State]:
        if row:
            states_of_interest = set(
                [
                    self.F_2_inv[(x, y)]
                    for y in range(self.s)
                    if (x, y) in self.F_2_inv.keys()
                ]
            )
        else:
            states_of_interest = set(
                [
                    self.F_2_inv[(y, x)]
                    for y in range(self.s)
                    if (y, x) in self.F_2_inv.keys()
                ]
            )

        return states_of_interest

    def predecessors(self, x: int, a: Sym, row: bool) -> Set[State]:
        states_of_interest = self.projection_to_states(row, x)

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

    def __call__(self, h: npt.NDArray[np.int_], a: Sym) -> npt.NDArray[np.int_]:
        """Computes the next hidden state given the current hidden state in 4 phases
        of the Dewdney RNN.
        """

        h = h[:-1]

        # Phase 1: compute the activations of the individual transversal factors, which
        # will next be combined (conjugated) into the activations of the transversals.
        h_factors = H(self.U_factors @ h + self.b_factors)
        # h_factors.shape = (2 * # of transversals across all rows/columns of M, 1)

        # Phase 2: compute the activations of the transversals, which will later be
        # combined with the row/column line activations.
        h_transversals = H(self.U_transversals @ h_factors + self.b_transversals)
        # h_transversals.shape = (# of transversals across all rows/columns of M, 1)

        # Phase 3: compute the activations of the row/column lines, which will later be
        # combined with the transversal activations to compute the new hidden state of
        # the RNN.
        h_non_transversals = H(self.U_non_transversals @ h + self.b_non_transversals)
        # h_non_transversals.shape = (# of rows/columns of M, 1)

        # Phase 4: compute the new activations of the state neurons in the new state
        # of the RNN.
        # Phase 4.1: combine the activations of the transversals with the row/column
        # line activations to compute the activations of all the transition neurons of
        # the coordinates of the representations.
        h_lines = np.concatenate((h_transversals, h_non_transversals), axis=0)

        # Phase 4.1: compute the activations of the state neurons for each possible
        # input symbol.
        h_cells = H(self.U_cells @ h_lines + self.b_cells)

        # input symbol combinations.
        h_states_alphabet = H(
            h_cells + self.V @ fsa_utils.sym_one_hot(a, self.M, self.sym2idx)
        )

        # Phase 4.2: compute the activations of the state neurons based on the actual
        # input symbol.
        h_states = H(self.U_states @ h_states_alphabet + self.b_states)

        # Phase 5: compute the acceptance bit of the input string so far
        accept = H(
            self.U_accept @ h_states
            + self.V_accept @ fsa_utils.sym_one_hot(a, self.M, self.sym2idx)
        )

        # Phase 6: combine the activations of the state neurons with the acceptance bit
        # to compute the new hidden state of the RNN.
        h = np.concatenate((h_states, accept), axis=0)

        return h
