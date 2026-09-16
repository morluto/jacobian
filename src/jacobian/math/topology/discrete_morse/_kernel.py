"""Exact bounded kernel for discrete Morse matching construction.

The kernel canonicalizes the source complex once, admits the resource
envelope, checks the supplied pairs against the complete face closure in
caller order, and decides acyclicity of the directed Hasse graph with one
iterative depth-first search that returns either a concrete closed V-path or
a topological order.  Validators never replay this mathematics.
"""

from __future__ import annotations

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    Simplex,
)
from jacobian.math.topology.discrete_morse._models import (
    MAX_MORSE_CELLS,
    MAX_MORSE_HASSE_EDGES,
    MAX_MORSE_PAIRS,
    CriticalCellProfile,
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
    MatchingPair,
    MorseMatchingFault,
    MorseMatchingOutcome,
)

_WHITE, _GRAY, _BLACK = 0, 1, 2


def _resource(
    code: str, message: str, location: tuple[str | int, ...]
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=f"topology.discrete_morse.{code}", message=message
    )


def _closure_cells(
    complex_: FiniteSimplicialComplex,
) -> tuple[tuple[Simplex, ...], ...]:
    """Return the face closure grouped by dimension in canonical order."""

    return tuple(tuple(group.faces) for group in complex_.faces_by_dimension)


def _cover_relations(
    cells: tuple[tuple[Simplex, ...], ...],
) -> list[tuple[Simplex, Simplex]]:
    """Return every codimension-one cover pair in canonical sorted order."""

    covers: list[tuple[Simplex, Simplex]] = []
    for dimension in range(len(cells) - 1):
        lower = set(cells[dimension])
        for coface in cells[dimension + 1]:
            for position in range(len(coface)):
                face = coface[:position] + coface[position + 1 :]
                if face in lower:
                    covers.append((face, coface))
    covers.sort()
    return covers


def _admit_envelope(
    cells: tuple[tuple[Simplex, ...], ...],
    request: DiscreteMorseMatchingRequest,
    covers: list[tuple[Simplex, Simplex]],
) -> int:
    """Check every public resource bound before matching work begins."""

    cell_count = sum(len(group) for group in cells)
    if cell_count > MAX_MORSE_CELLS:
        raise _resource(
            "admission.cells",
            f"the face closure has {cell_count} cells, above the "
            f"{MAX_MORSE_CELLS}-cell matching envelope",
            ("complex",),
        )
    if len(request.pairs) > MAX_MORSE_PAIRS:
        raise _resource(
            "admission.pairs",
            f"the matching has {len(request.pairs)} pairs, above the "
            f"{MAX_MORSE_PAIRS}-pair envelope",
            ("pairs",),
        )
    if len(covers) > MAX_MORSE_HASSE_EDGES:
        raise _resource(
            "admission.hasse_edges",
            f"the Hasse diagram has {len(covers)} cover edges, above the "
            f"{MAX_MORSE_HASSE_EDGES}-edge envelope",
            ("complex",),
        )
    return cell_count


def _invalid(
    complex_: FiniteSimplicialComplex,
    fault: MorseMatchingFault,
    message: str,
    position: int,
) -> DiscreteMorseMatchingResult:
    return DiscreteMorseMatchingResult._from_kernel(
        outcome=MorseMatchingOutcome.INVALID_MATCHING,
        complex=complex_,
        pairs=(),
        critical_profile=None,
        topological_order=(),
        hasse_edges=None,
        closed_v_path=(),
        fault=fault,
        fault_message=message,
        fault_pair_index=position,
    )


def _check_pairs(
    complex_: FiniteSimplicialComplex,
    request: DiscreteMorseMatchingRequest,
    covers: list[tuple[Simplex, Simplex]],
    cells: tuple[tuple[Simplex, ...], ...],
) -> (
    tuple[tuple[MatchingPair, ...], dict[Simplex, Simplex], set[Simplex]]
    | DiscreteMorseMatchingResult
):
    """Validate the supplied pairs in caller order or return the first fault."""

    known: set[Simplex] = set()
    for group in cells:
        known.update(group)
    cover_set = set(covers)
    matched: dict[Simplex, Simplex] = {}
    used: set[Simplex] = set()
    accepted: list[tuple[Simplex, Simplex]] = []
    for position, pair in enumerate(request.pairs):
        face: Simplex = tuple(sorted(pair.face))
        coface: Simplex = tuple(sorted(pair.coface))
        for cell in (face, coface):
            if cell not in known:
                return _invalid(
                    complex_,
                    MorseMatchingFault.UNKNOWN_CELL,
                    f"cell {list(cell)} of pair {position} is not a face of the "
                    "source complex closure",
                    position,
                )
        if (face, coface) not in cover_set:
            return _invalid(
                complex_,
                MorseMatchingFault.NOT_A_COVER_PAIR,
                f"pair {position} ({list(face)}, {list(coface)}) is not a "
                "codimension-one cover relation of the face poset",
                position,
            )
        for cell in (face, coface):
            if cell in used:
                return _invalid(
                    complex_,
                    MorseMatchingFault.DUPLICATE_CELL,
                    f"cell {list(cell)} of pair {position} already occurs in an "
                    "earlier matching pair",
                    position,
                )
            used.add(cell)
        matched[face] = coface
        accepted.append((face, coface))
    accepted.sort()
    canonical_pairs = tuple(
        MatchingPair(face=face, coface=coface) for face, coface in accepted
    )
    return canonical_pairs, matched, used


def _iterative_cycle_or_order(
    node_count: int, adjacency: tuple[list[int], ...]
) -> tuple[list[int] | None, list[int] | None]:
    """Return one directed cycle, or ``None`` and a topological order.

    A cycle is returned as the ordered list of distinct nodes whose
    consecutive edges close back to the first node.
    """

    color = [_WHITE] * node_count
    parent = [-1] * node_count
    postorder: list[int] = []
    for start in range(node_count):
        if color[start] != _WHITE:
            continue
        color[start] = _GRAY
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node, position = stack[-1]
            if position < len(adjacency[node]):
                stack[-1] = (node, position + 1)
                successor = adjacency[node][position]
                if color[successor] == _GRAY:
                    cycle = [successor]
                    walker = node
                    while walker != successor:
                        cycle.append(walker)
                        walker = parent[walker]
                    cycle.reverse()
                    return cycle, None
                if color[successor] == _WHITE:
                    color[successor] = _GRAY
                    parent[successor] = node
                    stack.append((successor, 0))
            else:
                color[node] = _BLACK
                postorder.append(node)
                stack.pop()
    postorder.reverse()
    return None, postorder


def _critical_profile(
    complex_: FiniteSimplicialComplex,
    cells: tuple[tuple[Simplex, ...], ...],
    used: set[Simplex],
) -> CriticalCellProfile:
    counts = [0] * len(cells)
    critical: list[Simplex] = []
    for dimension, group in enumerate(cells):
        for face in group:
            if face not in used:
                counts[dimension] += 1
                critical.append(face)
    return CriticalCellProfile(
        counts_by_dimension=tuple(counts),
        critical_cells=tuple(critical),
        euler_characteristic=sum(
            (-1) ** dimension * count for dimension, count in enumerate(counts)
        ),
        closure_euler_characteristic=sum(
            (-1) ** dimension * count
            for dimension, count in enumerate(complex_.f_vector)
        ),
    )


def construct_matching(
    complex_: FiniteSimplicialComplex,
    request: DiscreteMorseMatchingRequest,
) -> DiscreteMorseMatchingResult:
    """Decide one supplied matching against the canonical source complex."""

    cells = _closure_cells(complex_)
    covers = _cover_relations(cells)
    cell_count = _admit_envelope(cells, request, covers)
    checked = _check_pairs(complex_, request, covers, cells)
    if isinstance(checked, DiscreteMorseMatchingResult):
        return checked
    canonical_pairs, matched, used = checked

    index_of: dict[Simplex, int] = {}
    flat: list[Simplex] = []
    for group in cells:
        for face in group:
            index_of[face] = len(flat)
            flat.append(face)
    adjacency: list[list[int]] = [[] for _ in flat]
    for face, coface in covers:
        if matched.get(face) == coface:
            adjacency[index_of[face]].append(index_of[coface])
        else:
            adjacency[index_of[coface]].append(index_of[face])

    cycle, order = _iterative_cycle_or_order(cell_count, tuple(adjacency))
    if cycle is not None:
        return DiscreteMorseMatchingResult._from_kernel(
            outcome=MorseMatchingOutcome.CYCLIC_MATCHING,
            complex=complex_,
            pairs=canonical_pairs,
            critical_profile=None,
            topological_order=(),
            hasse_edges=None,
            closed_v_path=tuple(flat[node] for node in cycle),
            fault=None,
            fault_message=None,
            fault_pair_index=None,
        )
    assert order is not None
    return DiscreteMorseMatchingResult._from_kernel(
        outcome=MorseMatchingOutcome.ACYCLIC_MATCHING,
        complex=complex_,
        pairs=canonical_pairs,
        critical_profile=_critical_profile(complex_, cells, used),
        topological_order=tuple(flat[node] for node in order),
        hasse_edges=len(covers),
        closed_v_path=(),
        fault=None,
        fault_message=None,
        fault_pair_index=None,
    )


__all__ = ["construct_matching"]
