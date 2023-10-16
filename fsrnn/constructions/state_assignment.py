from itertools import product
from math import ceil, sqrt
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from fsrnn.base.symbol import Sym


def _find_transversal(B: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Finds the transversal with the most ones in the matrix B.

    Args:
        B (npt.NDArray[np.int_]): The matrix whose transversal is to be found.

    Returns:
        npt.NDArray[np.int_]: The transversal with the most ones in B,
                              represented as a matrix.
    """
    assert np.unique(B).tolist() == [0, 1]
    B = csr_matrix(B, dtype=np.int_)
    perm = maximum_bipartite_matching(B, perm_type="column")

    L = np.eye(B.shape[0], dtype=np.int_)[:, perm].T
    L[perm == -1, :] = np.zeros((np.sum(perm == -1), B.shape[1]), dtype=np.int_)

    return L


def _find_non_tranversals(B: npt.NDArray[np.int_]) -> List[npt.NDArray[np.int_]]:
    """Given the matrix B with the maximal matching of at most 2*N - n ones, it finds
    the corresponding cover (set of rows or columns) that cover the remaining ones.
    (Row or columns are chosen based on which ones actually cover the remaining ones.
     Since the rows and columns form a bipartite graph,
     only one of those has to be used.)

    Args:
        B (npt.NDArray[np.int_]): The matrix whose cover is to be found.

    Returns:
        List[npt.NDArray[np.int_]]: The cover of B.
    """
    n_row_ones, n_col_ones = np.max(B, axis=1).sum(), np.max(B, axis=0).sum()

    non_transversals = []
    if n_row_ones < n_col_ones:
        for ii in range(B.shape[0]):
            if np.sum(B[ii, :] == 1) > 0:  # If this row has ones
                line = np.zeros(B.shape, dtype=np.int_)
                line[ii, :] = B[ii, :]
                non_transversals.append(line)
    else:
        for jj in range(B.shape[1]):
            if np.sum(B[:, jj] == 1) > 0:
                line = np.zeros(B.shape, dtype=np.int_)
                line[:, jj] = B[:, jj]
                non_transversals.append(line)

    return non_transversals


def cover(B: npt.NDArray[np.int_]) -> Tuple[List[npt.NDArray[np.int_]], int]:
    """Constructs the cover of the matrix B with lines. A line is a matrix with either
    a single row in which there are consecutive ones, or a single column in which there
    are consecutive ones, or a transversal, i.e., a matrix with ones such that there
    are no ones in the same row or column. The cover is a list of lines such that the
    union of the lines is the matrix B.

    Args:
        B (npt.NDArray[np.int_]): The matrix to be covered.

    Returns:
        List[npt.NDArray[np.int_]]: The cover of B.
    """
    assert np.unique(B).tolist() == [0, 1]

    n_ones = np.sum(B == 1)
    n_transversals, N = 0, ceil(sqrt(n_ones))
    n_transversals = 0

    lines = []
    B_n = B.copy()
    while np.sum(B_n == 1) > 0:
        line = _find_transversal(B_n)
        if np.sum(line == 1) <= 2 * N - n_transversals:
            # if np.sum(line == 1) <= 0:
            # If the maximum matching is of size less than or equal to 2N - n, then
            # there exists a cover with at most 2*N-n "vertices" (rows/columns).
            # In fact, as long as we stop producing transversals at some point when
            # #ones <= (1 + ε)*N - n, we can find a cover with at most O(N)).
            non_tranversal_lines = _find_non_tranversals(B_n)
            lines.extend(non_tranversal_lines)
            assert (B_n == np.sum(non_tranversal_lines, axis=0)).all()
            break
        else:
            lines.append(line)
            B_n = B_n - line
            n_transversals += 1

    return lines, n_transversals


def transversal_factorization(
    L: npt.NDArray[np.int_],
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Computes the factorization of the transversal L into two matrices L_1 and L_2 such
    that L = L_1 and L_2.

    The factors L_1 and L_2 are lower and upper triangular matrices, respectively.
    Together they form a diagonal matrix, which is a permutation of the original
    transversal.

    Args:
        L (npt.NDArray[np.int_]): The transversal to be factored.

    Returns:
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: The factors of L.
    """

    D = L @ L.T
    L_1, L_2 = np.tril(np.ones(L.shape, dtype=np.int_)), np.triu(
        np.ones(L.shape, dtype=np.int_)
    )

    zero_out = np.diag(D) == 0
    L_1[zero_out, zero_out] = L_2[zero_out, zero_out] = 0

    return L_1, L_2


def factor_northwest_permutation(F: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Computes the permutation matrix required to make the transversal factor matrix F
    a northwest matrix, i.e., a matrix in which all ones are in the northwest corner.

    Args:
        F (npt.NDArray[np.int_]): The factor to be permuted.

    Returns:
        npt.NDArray[np.int_]: The permutation matrix.
    """
    P = np.eye(F.shape[0], dtype=np.int_)[range(F.shape[0] - 1, -1, -1), :]

    return P


def non_transversal_permutation(
    L: npt.NDArray[np.int_], row: bool
) -> npt.NDArray[np.int_]:
    """Computes the permutation matrix to make the non-transversal line L (i.e., a
    row or a column matrix) a northwest matrix.
    The row or column could have non-contiguous stretches of ones.

    Args:
        L (npt.NDArray[np.int_]): The non-tranversal line to be permuted.
        row (bool): Whether L is a row or a column.

    Returns:
        npt.NDArray[np.int_]: The permutation matrix.
    """
    P = np.zeros(L.shape, dtype=np.int_)
    if row:
        P[np.where(np.max(L, axis=0) == 1)[0], range(np.sum(L == 1))] = 1
    else:
        # P[np.where(np.max(L, axis=1) == 1)[0], range(np.sum(L == 1))] = 1
        P[range(np.sum(L == 1)), np.where(np.max(L, axis=1) == 1)[0]] = 1

    return P


def non_transversal_northwest_permutation(
    L: npt.NDArray[np.int_], row: bool
) -> npt.NDArray[np.int_]:
    """Computes the permutation matrix to make the non-transversal line L (i.e., a
    row or a column matrix) a northwest matrix after its ones have been moved into
    a contiguous block (see `non_transversal_permutation`).

    Args:
        L (npt.NDArray[np.int_]): The non-transversal line (row or column matrix)
                                  to be permuted.
        row (bool): Whether L is a row or a column.

    Returns:
        npt.NDArray[np.int_]: The permutation matrix.
    """
    P = np.zeros(L.shape, dtype=np.int_)

    if row:
        # Since we do this after ensuring that the ones are contiguous, we can just
        # look at the first column.
        idx = np.argmax(L, axis=0)[0]
    else:
        # Same as above.
        idx = np.argmax(L, axis=1)[0]

    P[idx, 0] = P[0, idx] = 1

    return P


def northwest_alpha(M: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Computes the α vector of the northwest matrix M from (5.3) in Alon et al.
    The values in the vector will be used as the weights of the RNN transition matrices.
    The values in the α vector together with the ones in the β vector make sure
    that only the entries in the northwest corner of the transition matrix are non-zero
    after applying the Heaviside function.
    See p. 238 in Dewdney (1977).

    Args:
        M (npt.NDArray[np.int_]): The northwest matrix.

    Returns:
        npt.NDArray[np.int_]: The α vector.
    """
    α = np.sum(M, axis=1)
    return α


def northwest_beta(F: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Computes the β vector of the northwest matrix F from (5.5) in Alon et al.
    The values in the vector will be used as the weights of the RNN transition matrices.
    See `northwest_alpha` for more details.

    Args:
        F (npt.NDArray[np.int_]): The northwest matrix.

    Returns:
        npt.NDArray[np.int_]: The β vector.
    """
    β = np.asarray(range(F.shape[0], 0, -1))
    return β


def build_state_set_detector(
    B: npt.NDArray[np.int_],
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    B is a binary matrix that encodes states in using a two-dimensional encoding.
    If enc(q) = [i, j], then B[i, j] = 1.
    This function constructs the matrix of vectors [α β] for detecting the activation
    of the representations of the states represented by 1's in B.
    That is, it constructs the vectors α and β such that the v_i * [α β] > D if and only
    if v_i encodes some state q_i in B, where D is the size of the matrix.
    These activations can then OR-ed to detect the activation of any state in B.
    If B is the predecessor matrix, this OR can be used to detect which states can be
    activated at a given time step.
    Based on equations 5.3 and 5.5 in Alon et al. (1991).

    Args:
        B (npt.NDArray[np.int_]): The binary matrix encoding the states whose
            activation should be detected.
            Usually, this is the predecessor matrix.

    Returns:
        npt.NDArray[np.int_]: The set of vectors [α β] that detect the activation of
        the states in B split into two parts: those that detect the activation of
        transversals (factorized into factors) and those that detect the activation of
        non-transversals.
    """

    D = 2 * B.shape[1]

    # This also works if the set states of interest (in the row/column k) is empty
    if np.sum(B) == 0:
        # There are no predecessors of the states in this row/column.
        return (
            np.zeros((0, D), dtype=np.int_),
            np.zeros((0, D), dtype=np.int_),
        )

    # Build the lines that cover the predecessor matrix
    lines, n_transversals = cover(B)

    ########################################
    # Build the state activation matrices ##
    ########################################

    # The matrix that will be used to compute the activations of the individual
    # factors in the transversal matrices
    U_factors = np.zeros((2 * n_transversals, D), dtype=np.int_)

    # The matrix that will compute the non-transversal line activations
    U_non_transversal = np.zeros((len(lines) - n_transversals, D), dtype=np.int_)

    for ii, L in enumerate(lines):
        # If the line is a transversal, we need to factor it
        # and construct the transition matrices for the two factors
        # individually, and then combine them by taking the logical and
        # implemented by the RNN update step.

        if ii < n_transversals:
            F_1, F_2 = transversal_factorization(L)

            P_1 = factor_northwest_permutation(F_1)
            P_2 = factor_northwest_permutation(F_2)

            # These will be the weights in the matrices for the two factors
            # By construction, the *rows* of the first factor are permuted, and the
            # *columns* of the second factor are permuted.
            α_1, β_1 = northwest_alpha(P_1 @ F_1), northwest_beta(F_1)
            α_2, β_2 = northwest_alpha(F_2 @ P_2), northwest_beta(F_2 @ P_2)

            # Build the state activation matrix for this specific transversal
            # The first s cells in the hidden state will correspond to the
            # row indices, which are weighted by the α vector
            # while the last s cells will correspond to the column indices,
            # which are weighted by the β vector.
            # Since the α and β vectors were computed based on the
            # *permuted* factor matrices (make the factor matrices northwest),
            # we permute them back.
            # We also need to permute the β vectors with the L.T matrix,
            # since the original line matrix was permuted with the L.T matrix,
            # which shuffled the columns of the transversal.
            U_factors[2 * ii, :] = np.concatenate((P_1 @ α_1, L.T @ β_1))
            U_factors[2 * ii + 1, :] = np.concatenate((α_2, L.T @ P_2 @ β_2))

        # If the line is not a transversal, we can compute the activations
        # directly by taking into account that this is a proto-corner matrix.
        else:
            line_is_row = np.max(L, axis=0).sum() > np.max(L, axis=1).sum()

            P_c = non_transversal_permutation(L, line_is_row)
            L_c = L @ P_c if line_is_row else P_c @ L

            # Construct the permutation matrix of the rows to move the row
            # with the most ones to the top (if row) or the left (if column)
            P_nw = non_transversal_northwest_permutation(L_c, row=line_is_row)

            # Compute the α and β vectors based on the permuted matrix
            M = P_nw @ L_c if line_is_row else L_c @ P_nw
            α, β = northwest_alpha(M), northwest_beta(M)

            # Permute the α and β vectors back to be applicable on the
            # original hidden state
            if line_is_row:
                α, β = P_nw.T @ α, P_c @ β
            else:
                α, β = P_c.T @ α, P_nw @ β

            # Put the α and β vectors in the state activation matrix, as
            # they will decide whether a specific two-dimensional state
            # representation is in the north-western corner of the csr_matrix
            # (after permutation).
            U_non_transversal[ii - n_transversals, :] = np.concatenate((α, β))

    return U_factors, U_non_transversal


def detect_transversal_factors(
    factor_matrices: List[npt.NDArray[np.int_]], s: int
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    # This matrix detect the activation of individual factors of the transversals
    U = np.vstack(factor_matrices)
    # The sum of the two coordinates must be at least s + 1
    b = -s * np.ones((U.shape[0]), dtype=np.int_)
    # U.shape = (# total transversals across M and Σ, D)

    return U, b


def detect_non_transversals(
    non_transversal_matrices: List[npt.NDArray[np.int_]], s: int
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    # This matrix detects the activation of the non-transversal lines,
    # i.e. the row and column matrices.`
    U = np.vstack(non_transversal_matrices)
    # The sum of the two coordinates must be at least s + 1
    b = -s * np.ones((U.shape[0]), dtype=np.int_)
    # U.shape =
    # (# total non-transversal lines across M and Σ, D)

    return U, b


def conjugate_factors(
    factor_matrices: List[npt.NDArray[np.int_]],
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    transversals_per_coordinate = [M.shape[0] // 2 for M in factor_matrices]
    total_n_transversals = sum(transversals_per_coordinate)

    # This matrix converts the activations of the transversal factors into
    # the activations of the transversals they represent.
    U = np.zeros((total_n_transversals, 2 * total_n_transversals), dtype=np.int_)
    # U.shape =
    # (# total transversals across M and Σ, 2 * # total transversals across M and Σ)

    # This creates a matrix which conjuncts subsequent 2 elements in the vector
    # with the stride 2.
    # It has the form:
    # [[1, 1, 0, 0, ...],
    #  [0, 0, 1, 1, ...],
    #  [0, 0, 0, 0, ...],
    U[:, range(0, 2 * total_n_transversals, 2)] = np.eye(total_n_transversals)
    U[:, range(1, 2 * total_n_transversals, 2)] = np.eye(total_n_transversals)

    # Both factors must be active, so the this ensures the transversal is active
    # if and only if both factors are active.
    b = -1 * np.ones((total_n_transversals,), dtype=np.int_)

    return U, b


def count_lines(
    factor_matrices: List[npt.NDArray[np.int_]],
    non_transversal_matrices: List[npt.NDArray[np.int_]],
) -> Tuple[List[int], List[int]]:
    transversals_per_coordinate = [U.shape[0] // 2 for U in factor_matrices]
    non_transversals_per_coordinate = [U.shape[0] for U in non_transversal_matrices]
    return transversals_per_coordinate, non_transversals_per_coordinate


def detect_input_independent_states(
    transversals_per_coordinate: List[int],
    non_transversals_per_coordinate: List[int],
    s: int,
    M: int,
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    assert len(transversals_per_coordinate) == len(non_transversals_per_coordinate)

    n_lines = len(transversals_per_coordinate) + len(non_transversals_per_coordinate)

    # This matrix detects the activation of the cells in the hidden state
    # corresponding to the encodings of the states of the automaton,
    # with one cell for each possible input symbol.
    # For each cell, the activation of any of its lines is sufficient,
    # so the U_cells matrix must be sensitive to all of them.
    # However, we again have construct it in a similar striding fashion
    # as the U_transversals matrix.
    U = np.zeros((M * 2 * s + M + 1, n_lines), dtype=np.int_)
    # The vector σ(U_cells @ h_prime + b_cells) will contain *two*
    # entries for each row/column (entry in the hidden state): one for each possible
    # input symbol.
    # Those two entries then have to be combined to get the final activation
    # of the state neuron based on the input symbol.

    C, t_c, nt_c = sum(transversals_per_coordinate), 0, 0
    for ii, (t_n, nt_n) in enumerate(
        zip(transversals_per_coordinate, non_transversals_per_coordinate)
    ):
        # This puts 1's in the cells which will detect the activation of the
        # transversal and non-transversal lines of the ii-th coordinate
        # (row or column of the state encoding matrix).
        # Recall that the vector transformer by this matrix if of the form:
        # [# total transversal lines across M, # total non-transversal lines]

        U[ii, t_c : t_c + t_n] = 1
        U[ii, C + nt_c : C + nt_c + nt_n] = 1
        t_c += t_n
        nt_c += nt_n

    # At least one line must be active, so we set the bias to 0
    b = np.zeros((M * 2 * s + M + 1,), dtype=np.int_)
    # U_cells.shape =
    # ((|Σ| + 1) * 2 * s + S + 1,
    # # total transversals + # total non-transversal lines),
    # with the last s + 1 dimensions corresponding to the
    # one-hot encoding of the input symbol.

    return U, b


def detect_next_state(
    Σ: List, s: int, M: int, sym2idx: Dict[Sym, int]
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    U = np.zeros((2 * s, 2 * s * M), dtype=np.int_)

    for a, k in product(Σ, range(s)):
        ix = sym2idx[a]
        U[k, ix * s + k] = 1
        U[s + k, M * s + ix * s + k] = 1

    # Since at most one symbol subvector can be active, we set the bias to 0
    b = np.zeros((2 * s,), dtype=np.int_)

    return U, b
