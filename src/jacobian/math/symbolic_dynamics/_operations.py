"""Domain-owned symbolic dynamics operations."""

from __future__ import annotations

import itertools
import math
from collections import deque

import networkx as nx

from jacobian.math.symbolic_dynamics._models import (
    AdjacencyShiftRequest,
    AdjacencyShiftResult,
    BlockLanguageRequest,
    BlockLanguageResult,
    FiniteTypeShiftRequest,
    FiniteTypeShiftResult,
    HigherBlockRequest,
    HigherBlockResult,
    PeriodicPointProfileRequest,
    PeriodicPointProfileResult,
)


def _contains(word: tuple, pattern: tuple) -> bool:
    """Check if pattern is a contiguous factor of word."""
    if len(pattern) > len(word):
        return False
    if len(pattern) == 0:
        return True
    for i in range(len(word) - len(pattern) + 1):
        if word[i : i + len(pattern)] == pattern:
            return True
    return False


def construct_finite_type_shift(request: FiniteTypeShiftRequest) -> FiniteTypeShiftResult:
    """Construct a shift of finite type from a forbidden-block family."""
    alphabet = request.alphabet
    forbidden = request.forbidden_blocks
    max_len = max((len(b) for b in forbidden), default=0)

    has_empty_forbidden = any(len(b) == 0 for b in forbidden)
    if has_empty_forbidden:
        return FiniteTypeShiftResult(
            alphabet=alphabet,
            forbidden_blocks=forbidden,
            max_forbidden_length=max_len,
            is_empty=True,
            adjacency_matrix=(),
            num_states=0,
        )

    minimal_forbidden = []
    for block in forbidden:
        is_redundant = False
        for other in forbidden:
            if other == block:
                continue
            if len(other) < len(block) and _contains(block, other):
                is_redundant = True
                break
        if not is_redundant:
            minimal_forbidden.append(block)

    if max_len <= 1:
        valid = set(alphabet)
        for block in minimal_forbidden:
            if len(block) == 1:
                valid.discard(block[0])
        n = len(valid)
        matrix = [[1] * n for _ in range(n)] if n > 0 else []
        return FiniteTypeShiftResult(
            alphabet=alphabet,
            forbidden_blocks=tuple(minimal_forbidden),
            max_forbidden_length=max_len,
            is_empty=n == 0,
            adjacency_matrix=tuple(tuple(r) for r in matrix),
            num_states=n,
        )

    states = list(itertools.product(alphabet, repeat=max_len - 1))
    state_to_idx = {s: i for i, s in enumerate(states)}
    forbidden_set = set(minimal_forbidden)
    n = len(states)
    matrix = [[0] * n for _ in range(n)]

    for i, state in enumerate(states):
        for letter in alphabet:
            new_block = state + (letter,)
            if new_block not in forbidden_set:
                next_state = new_block[1:]
                j = state_to_idx[next_state]
                matrix[i][j] += 1

    return FiniteTypeShiftResult(
        alphabet=alphabet,
        forbidden_blocks=tuple(minimal_forbidden),
        max_forbidden_length=max_len,
        is_empty=n == 0,
        adjacency_matrix=tuple(tuple(r) for r in matrix),
        num_states=n,
    )


def compute_block_language(request: BlockLanguageRequest) -> BlockLanguageResult:
    """Compute the allowed block language of a shift at a given length."""
    alphabet = request.alphabet
    forbidden = set(request.forbidden_blocks)
    block_length = request.block_length

    allowed: list[tuple[str, ...]] = []
    for word in itertools.product(alphabet, repeat=block_length):
        is_allowed = True
        for block in forbidden:
            if len(block) <= block_length and _contains(word, block):
                is_allowed = False
                break
        if is_allowed:
            allowed.append(word)

    return BlockLanguageResult(
        block_length=block_length,
        allowed_blocks=tuple(allowed),
        count=len(allowed),
    )


def construct_adjacency_shift(request: AdjacencyShiftRequest) -> AdjacencyShiftResult:
    """Construct a shift presentation from an adjacency matrix."""
    matrix = request.matrix
    n = len(matrix)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                graph.add_edge(i, j)

    is_essential = True
    for i in range(n):
        if graph.out_degree(i) == 0 or graph.in_degree(i) == 0:
            is_essential = False
            break

    is_irreducible = nx.is_strongly_connected(graph) if n > 0 else False

    period = 0
    if is_irreducible and n > 0:
        visited = {0}
        level = {0: 0}
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in graph.successors(u):
                if v not in visited:
                    visited.add(v)
                    level[v] = level[u] + 1
                    queue.append(v)
                else:
                    cycle_len = abs(level[u] + 1 - level[v])
                    if cycle_len > 0:
                        period = math.gcd(period, cycle_len) if period else cycle_len

    is_mixing = is_irreducible and period == 1

    return AdjacencyShiftResult(
        matrix=matrix,
        is_essential=is_essential,
        is_irreducible=is_irreducible,
        period=period,
        is_mixing=is_mixing,
    )


def compute_periodic_point_profile(
    request: PeriodicPointProfileRequest,
) -> PeriodicPointProfileResult:
    """Compute the periodic point profile of a shift."""
    matrix = request.matrix
    max_period = request.max_period
    n = len(matrix)

    def mat_mult(a, b):
        return [
            [sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]

    fix_counts: list[int] = []
    current = [list(row) for row in matrix]
    for k in range(1, max_period + 1):
        trace = sum(current[i][i] for i in range(n))
        fix_counts.append(trace)
        if k < max_period:
            current = mat_mult(current, matrix)

    def mobius_mu(n: int) -> int:
        if n == 1:
            return 1
        factors = set()
        d = n
        p = 2
        while p * p <= d:
            if d % p == 0:
                if p in factors:
                    return 0
                factors.add(p)
                d //= p
                while d % p == 0:
                    d //= p
                    if d % p == 0:
                        return 0
                p = 2
            else:
                p += 1
        if d > 1:
            if d in factors:
                return 0
            factors.add(d)
        return (-1) ** len(factors)

    exact_counts: list[int] = []
    for n_val in range(1, max_period + 1):
        total = 0
        for d in range(1, n_val + 1):
            if n_val % d == 0:
                mu = mobius_mu(n_val // d)
                total += mu * fix_counts[d - 1]
        exact_counts.append(total)

    orbit_counts = [exact_counts[i] // (i + 1) for i in range(max_period)]

    return PeriodicPointProfileResult(
        fix_counts=tuple(fix_counts),
        exact_counts=tuple(exact_counts),
        orbit_counts=tuple(orbit_counts),
        zeta_numerator=tuple(fix_counts),
        zeta_denominator=(1,) + tuple(0 for _ in range(max_period - 1)),
    )


def compute_higher_block(request: HigherBlockRequest) -> HigherBlockResult:
    """Compute the n-th higher-block presentation of a shift."""
    alphabet = request.alphabet
    n = request.n

    new_alphabet = ["".join(a) for a in itertools.product(alphabet, repeat=n - 1)]

    new_forbidden: list[tuple[str, ...]] = []
    for block in request.forbidden_blocks:
        if len(block) >= n:
            new_forbidden.append(tuple("".join(b) for b in zip(*[block[i:] for i in range(n - 1)])))

    return HigherBlockResult(
        new_alphabet=tuple(new_alphabet),
        new_forbidden_blocks=tuple(new_forbidden),
        n=n,
    )


__all__ = [
    "compute_block_language",
    "compute_higher_block",
    "compute_periodic_point_profile",
    "construct_adjacency_shift",
    "construct_finite_type_shift",
]
