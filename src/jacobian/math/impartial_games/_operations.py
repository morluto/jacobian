"""Domain-owned impartial combinatorial game operations."""

from __future__ import annotations

from functools import reduce
from operator import xor

import networkx as nx

from jacobian.math.impartial_games._models import (
    BirthdayRequest,
    BirthdayResult,
    GameMove,
    GrundyClassEntry,
    GrundyClassesRequest,
    GrundyClassesResult,
    GrundyEntry,
    GrundyTableRequest,
    GrundyTableResult,
    ImpartialGameDAGRequest,
    MexRequest,
    MexResult,
    NimEquivalentRequest,
    NimEquivalentResult,
    NimOption,
    NimOptionsRequest,
    NimOptionsResult,
    NimSumRequest,
    NimSumResult,
    OutcomeProfileRequest,
    OutcomeProfileResult,
    PositionGrundyRequest,
    PositionGrundyResult,
    SubtractionDAGRequest,
    SubtractionDAGResult,
    SubtractionGrundyPrefixRequest,
    SubtractionGrundyPrefixResult,
)


def _build_digraph(game: ImpartialGameDAGRequest) -> nx.DiGraph:
    """Build a NetworkX digraph from the game DAG request."""
    graph = nx.DiGraph()
    graph.add_nodes_from(game.positions)
    for move in game.moves:
        graph.add_edge(move.source, move.target)
    return graph


def _check_acyclic(graph: nx.DiGraph) -> None:
    """Reject cyclic game DAGs."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("game DAG must be acyclic; cycles are not allowed")


def _topological_order(graph: nx.DiGraph) -> list[str]:
    """Return a deterministic topological order of the graph nodes."""
    return list(nx.topological_sort(graph))


def _compute_grundy_table(
    graph: nx.DiGraph,
) -> tuple[dict[str, int], dict[str, tuple[int, ...]], list[str]]:
    """Compute Grundy values for all positions in reverse topological order.

    Terminal positions (no successors) have Grundy value 0.  For each
    non-terminal position, the Grundy value is the mex of its successors'
    Grundy values.  Processing in reverse topological order ensures
    all successors are computed before the position itself.

    Returns (grundy_map, option_grundy_sets, topo_order).
    """
    topo = _topological_order(graph)
    # Reverse: process terminal positions first
    grundy: dict[str, int] = {}
    option_sets: dict[str, tuple[int, ...]] = {}

    for pos in reversed(topo):
        successors = list(graph.successors(pos))
        opt_values = sorted(grundy[s] for s in successors)
        option_sets[pos] = tuple(opt_values)
        grundy[pos] = _mex_of_set(opt_values)

    return grundy, option_sets, topo


def _mex_of_set(values: list[int]) -> int:
    """Compute the mex of a sorted list of non-negative integers."""
    mex = 0
    for v in values:
        if v == mex:
            mex += 1
        elif v > mex:
            break
    return mex


def compute_mex(request: MexRequest) -> MexResult:
    """Compute the minimum excluded value of a bounded finite set."""
    values = sorted(set(request.values))
    mex = 0
    membership = []
    for v in values:
        if v == mex:
            membership.append(mex)
            mex += 1
        elif v > mex:
            break
    return MexResult(
        mex=mex,
        membership_prefix=tuple(membership),
        first_gap=mex,
    )


def compute_grundy_table(request: GrundyTableRequest) -> GrundyTableResult:
    """Compute the complete Grundy table of a finite impartial game DAG."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    grundy, option_sets, topo = _compute_grundy_table(graph)

    max_g = max(grundy.values()) if grundy else 0
    histogram = [0] * (max_g + 1)
    for v in grundy.values():
        histogram[v] += 1

    entries = tuple(
        GrundyEntry(
            position=pos,
            grundy=grundy[pos],
            option_grundy_set=option_sets[pos],
        )
        for pos in request.game.positions
    )

    return GrundyTableResult(
        entry_map=entries,
        max_grundy=max_g,
        histogram=tuple(histogram),
        topological_order=tuple(topo),
    )


def compute_position_grundy(request: PositionGrundyRequest) -> PositionGrundyResult:
    """Compute the Grundy value of a single position."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    target = request.position
    reachable = set(nx.descendants(graph, target))
    reachable.add(target)
    subgraph = graph.subgraph(reachable)
    topo = _topological_order(subgraph)

    grundy: dict[str, int] = {}
    for pos in reversed(topo):
        successors = list(subgraph.successors(pos))
        opt_values = sorted(grundy[s] for s in successors)
        grundy[pos] = _mex_of_set(opt_values)

    opt_grundy_set = sorted(
        grundy[s] for s in subgraph.successors(target)
    )

    return PositionGrundyResult(
        position=target,
        grundy=grundy[target],
        reachable_positions=tuple(topo),
        topological_order=tuple(topo),
        option_grundy_set=tuple(opt_grundy_set),
    )


def compute_outcome_profile(request: OutcomeProfileRequest) -> OutcomeProfileResult:
    """Compute the P/N partition of a finite impartial game."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    grundy, _, _ = _compute_grundy_table(graph)

    p_positions = tuple(p for p in request.game.positions if grundy[p] == 0)
    n_positions = tuple(p for p in request.game.positions if grundy[p] > 0)
    terminal = tuple(
        p for p in request.game.positions if graph.out_degree(p) == 0
    )

    grundy_map = tuple(
        (p, grundy[p]) for p in request.game.positions
    )

    return OutcomeProfileResult(
        p_positions=p_positions,
        n_positions=n_positions,
        terminal_positions=terminal,
        grundy_map=grundy_map,
    )


def compute_nim_equivalent(request: NimEquivalentRequest) -> NimEquivalentResult:
    """Find the canonical Nim heap equivalent to one position."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    grundy, _, _ = _compute_grundy_table(graph)

    return NimEquivalentResult(
        position=request.position,
        heap_size=grundy[request.position],
    )


def compute_grundy_classes(request: GrundyClassesRequest) -> GrundyClassesResult:
    """Partition positions by equal Grundy value."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    grundy, _, _ = _compute_grundy_table(graph)

    classes: dict[int, list[str]] = {}
    for pos in request.game.positions:
        g = grundy[pos]
        classes.setdefault(g, []).append(pos)

    max_g = max(grundy.values()) if grundy else 0
    histogram = [0] * (max_g + 1)
    for v in grundy.values():
        histogram[v] += 1

    sorted_classes = sorted(classes.items())
    return GrundyClassesResult(
        classes=tuple(
            GrundyClassEntry(grundy=g, positions=tuple(sorted(positions)))
            for g, positions in sorted_classes
        ),
        histogram=tuple(histogram),
    )


def compute_birthday(request: BirthdayRequest) -> BirthdayResult:
    """Compute the birthday (DAG height) of every position."""
    graph = _build_digraph(request.game)
    _check_acyclic(graph)

    topo = _topological_order(graph)
    birthday: dict[str, int] = {}

    for pos in reversed(topo):
        successors = list(graph.successors(pos))
        if not successors:
            birthday[pos] = 0
        else:
            birthday[pos] = 1 + max(birthday[s] for s in successors)

    return BirthdayResult(
        birthdays=tuple((p, birthday[p]) for p in request.game.positions)
    )


# -- Nim operations ---------------------------------------------------------


def compute_nim_sum(request: NimSumRequest) -> NimSumResult:
    """Compute the nim-sum (bitwise xor) of heap sizes."""
    total = reduce(xor, request.heaps, 0)
    return NimSumResult(
        heaps=request.heaps,
        nim_sum=total,
        is_p_position=(total == 0),
    )


def compute_nim_options(request: NimOptionsRequest) -> NimOptionsResult:
    """Enumerate all legal options of a Nim position."""
    heaps = request.heaps
    options: list[NimOption] = []
    for i, size in enumerate(heaps):
        for new_size in range(size):
            resulting = list(heaps)
            resulting[i] = new_size
            options.append(
                NimOption(
                    heap_index=i,
                    old_size=size,
                    new_size=new_size,
                    resulting_heaps=tuple(resulting),
                )
            )
    return NimOptionsResult(options=tuple(options))


# -- Subtraction games ------------------------------------------------------


def compute_subtraction_dag(request: SubtractionDAGRequest) -> SubtractionDAGResult:
    """Build the game DAG for a bounded subtraction game."""
    s = sorted(request.subtraction_set)
    n = request.max_heap
    positions = tuple(str(i) for i in range(n + 1))
    moves: list[GameMove] = []
    for pos in range(n + 1):
        for sub in s:
            target = pos - sub
            if target >= 0:
                moves.append(
                    GameMove(source=str(pos), target=str(target))
                )
    terminal = tuple(
        str(i) for i in range(n + 1) if not any(i >= s for s in request.subtraction_set)
    )
    return SubtractionDAGResult(
        positions=positions,
        moves=tuple(moves),
        terminal_positions=terminal,
    )


def compute_subtraction_grundy_prefix(
    request: SubtractionGrundyPrefixRequest,
) -> SubtractionGrundyPrefixResult:
    """Compute the Grundy prefix of a bounded subtraction game."""
    s = sorted(request.subtraction_set)
    n = request.max_heap
    grundy = [0] * (n + 1)
    option_sets_list: list[tuple[int, ...]] = [()] * (n + 1)

    for pos in range(n + 1):
        opt_grundy: list[int] = []
        for sub in s:
            target = pos - sub
            if target >= 0:
                opt_grundy.append(grundy[target])
        opt_grundy.sort()
        option_sets_list[pos] = tuple(opt_grundy)
        grundy[pos] = _mex_of_set(opt_grundy)

    p_positions = tuple(i for i in range(n + 1) if grundy[i] == 0)
    n_positions = tuple(i for i in range(n + 1) if grundy[i] > 0)

    return SubtractionGrundyPrefixResult(
        grundy_values=tuple(grundy),
        option_sets=tuple(option_sets_list),
        p_positions=p_positions,
        n_positions=n_positions,
    )


__all__ = [
    "compute_birthday",
    "compute_grundy_classes",
    "compute_grundy_table",
    "compute_mex",
    "compute_nim_equivalent",
    "compute_nim_options",
    "compute_nim_sum",
    "compute_outcome_profile",
    "compute_position_grundy",
    "compute_subtraction_dag",
    "compute_subtraction_grundy_prefix",
]
