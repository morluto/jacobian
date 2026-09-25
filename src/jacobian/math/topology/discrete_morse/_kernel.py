"""Exact bounded kernel for discrete Morse matching construction.

The kernel canonicalizes the source complex once, admits the resource
envelope, checks the supplied pairs against the complete face closure in
caller order, and decides acyclicity of the directed Hasse graph with one
iterative depth-first search that returns either a concrete closed V-path or
a topological order.  Validators never replay this mathematics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.finite_fields import linear_algebra as prime_field
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    Simplex,
)
from jacobian.math.topology.discrete_morse._models import (
    MAX_MORSE_BOUNDARY_ENTRIES,
    MAX_MORSE_CELLS,
    MAX_MORSE_CRITICAL_CELLS,
    MAX_MORSE_GRADIENT_PATHS,
    MAX_MORSE_GRADIENT_STATES,
    MAX_MORSE_HASSE_EDGES,
    MAX_MORSE_PAIRS,
    MAX_MORSE_PATH_STEPS,
    CriticalCellBasis,
    CriticalCellProfile,
    DiscreteMorseMatchingResult,
    GradientPath,
    GradientPathCount,
    GradientPathsResult,
    GradientPathStep,
    MatchingPair,
    MorseBoundaryEntry,
    MorseComplexResult,
    MorseGradientStepKind,
    MorseMatchingFault,
    MorseMatchingOutcome,
)

_WHITE, _GRAY, _BLACK = 0, 1, 2
_DOWN = MorseGradientStepKind.DOWN
_UP = MorseGradientStepKind.UP


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

    if not complex_.faces_by_dimension:
        return ((),)
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
    pairs: tuple[MatchingPair, ...],
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
    if len(pairs) > MAX_MORSE_PAIRS:
        raise _resource(
            "admission.pairs",
            f"the matching has {len(pairs)} pairs, above the "
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
    pairs: tuple[MatchingPair, ...],
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
    for position, pair in enumerate(pairs):
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


@dataclass(frozen=True, slots=True)
class _DirectedCycle:
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TopologicalOrder:
    nodes: tuple[int, ...]


type _DirectedTraversal = _DirectedCycle | _TopologicalOrder


def _iterative_cycle_or_order(
    node_count: int, adjacency: tuple[list[int], ...]
) -> _DirectedTraversal:
    """Return one directed cycle or a topological order.

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
                    return _DirectedCycle(nodes=tuple(cycle))
                if color[successor] == _WHITE:
                    color[successor] = _GRAY
                    parent[successor] = node
                    stack.append((successor, 0))
            else:
                color[node] = _BLACK
                postorder.append(node)
                stack.pop()
    postorder.reverse()
    return _TopologicalOrder(nodes=tuple(postorder))


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
    pairs: tuple[MatchingPair, ...],
) -> DiscreteMorseMatchingResult:
    """Decide one supplied matching against the canonical source complex."""

    cells = _closure_cells(complex_)
    covers = _cover_relations(cells)
    cell_count = _admit_envelope(cells, pairs, covers)
    checked = _check_pairs(complex_, pairs, covers, cells)
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

    traversal = _iterative_cycle_or_order(cell_count, tuple(adjacency))
    if isinstance(traversal, _DirectedCycle):
        return DiscreteMorseMatchingResult._from_kernel(
            outcome=MorseMatchingOutcome.CYCLIC_MATCHING,
            complex=complex_,
            pairs=canonical_pairs,
            critical_profile=None,
            topological_order=(),
            hasse_edges=None,
            closed_v_path=tuple(flat[node] for node in traversal.nodes),
            fault=None,
            fault_message=None,
            fault_pair_index=None,
        )
    return DiscreteMorseMatchingResult._from_kernel(
        outcome=MorseMatchingOutcome.ACYCLIC_MATCHING,
        complex=complex_,
        pairs=canonical_pairs,
        critical_profile=_critical_profile(complex_, cells, used),
        topological_order=tuple(flat[node] for node in traversal.nodes),
        hasse_edges=len(covers),
        closed_v_path=(),
        fault=None,
        fault_message=None,
        fault_pair_index=None,
    )


def _faces_by_coface(
    cells: tuple[tuple[Simplex, ...], ...],
) -> dict[Simplex, tuple[Simplex, ...]]:
    """Return every codimension-one face of each cell in the face closure."""

    faces: dict[Simplex, tuple[Simplex, ...]] = {}
    for dimension in range(1, len(cells)):
        lower = set(cells[dimension - 1])
        for coface in cells[dimension]:
            local: list[Simplex] = []
            for position in range(len(coface)):
                face = coface[:position] + coface[position + 1 :]
                if face in lower:
                    local.append(face)
            faces[coface] = tuple(sorted(local))
    return faces


def _matching_maps(
    pairs: tuple[MatchingPair, ...],
) -> tuple[dict[Simplex, Simplex], dict[Simplex, Simplex]]:
    """Return the canonical lower-to-upper and upper-to-lower matching maps."""

    matched = {tuple(sorted(pair.face)): tuple(sorted(pair.coface)) for pair in pairs}
    upper = {coface: face for face, coface in matched.items()}
    return matched, upper


@dataclass(frozen=True, slots=True)
class _AdmittedAcyclicMatching:
    result: DiscreteMorseMatchingResult
    critical_profile: CriticalCellProfile


def _admit_acyclic_matching(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> _AdmittedAcyclicMatching:
    """Establish the supplied matching is acyclic before Morse reduction."""

    matching = construct_matching(complex_, pairs)
    if matching.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING:
        profile = matching.critical_profile
        if profile is None:
            raise RuntimeError("acyclic matching has no critical-cell profile")
        return _AdmittedAcyclicMatching(result=matching, critical_profile=profile)
    if matching.outcome is MorseMatchingOutcome.CYCLIC_MATCHING:
        detail = (
            "the supplied matching contains the closed V-path "
            f"{[list(cell) for cell in matching.closed_v_path]}"
        )
    else:
        detail = matching.fault_message or "the supplied matching is invalid"
    raise OperationDomainValidationError(
        location=("pairs",),
        code=(
            "topology.discrete_morse.matching_not_acyclic."
            f"{matching.outcome.value.lower()}"
        ),
        message=f"Morse reduction requires an acyclic matching: {detail}",
    )


def _enumerate_gradient_paths(
    faces_of: dict[Simplex, tuple[Simplex, ...]],
    matched: dict[Simplex, Simplex],
    upper: dict[Simplex, Simplex],
    critical: set[Simplex],
    start: Simplex,
    target: Simplex | None,
    *,
    max_paths: int,
    max_states: int,
    max_steps: int,
) -> tuple[list[tuple[tuple[MorseGradientStepKind, Simplex, Simplex], ...]], int]:
    """Enumerate every complete gradient path from one critical cell.

    A path alternates ``DOWN`` unmatched covers and ``UP`` matched covers,
    starts and ends with a ``DOWN`` step, and terminates at a critical cell.
    The complete family is bounded by the remaining path, state, and step
    budgets; an overflow is a resource rejection, never a truncated family.
    """

    paths: list[tuple[tuple[MorseGradientStepKind, Simplex, Simplex], ...]] = []
    states = 0
    stack: list[
        tuple[Simplex, tuple[tuple[MorseGradientStepKind, Simplex, Simplex], ...]]
    ] = [(start, ())]
    while stack:
        current, steps = stack.pop()
        states += 1
        if states > max_states:
            raise _resource(
                "admission.gradient_path_states",
                "gradient-path search exceeds the "
                f"{MAX_MORSE_GRADIENT_STATES}-state envelope",
                ("pairs",),
            )
        for face in faces_of.get(current, ()):
            if upper.get(current) == face:
                continue
            if len(steps) + 1 > max_steps:
                raise _resource(
                    "admission.gradient_path_length",
                    f"a gradient path exceeds the {MAX_MORSE_PATH_STEPS}-step envelope",
                    ("pairs",),
                )
            if face in critical:
                if target is None or face == target:
                    paths.append((*steps, (_DOWN, current, face)))
                    if len(paths) > max_paths:
                        raise _resource(
                            "admission.gradient_paths",
                            "the gradient-path family exceeds the "
                            f"{MAX_MORSE_GRADIENT_PATHS}-path envelope",
                            ("pairs",),
                        )
                continue
            successor = matched.get(face)
            if successor is None:
                continue
            stack.append(
                (
                    successor,
                    (*steps, (_DOWN, current, face), (_UP, face, successor)),
                )
            )
    return paths, states


def _gradient_path(
    steps: tuple[tuple[MorseGradientStepKind, Simplex, Simplex], ...],
) -> GradientPath:
    return GradientPath(
        start=steps[0][1],
        target=steps[-1][2],
        steps=tuple(
            GradientPathStep(kind=kind, source=source, target=target)
            for kind, source, target in steps
        ),
    )


def compute_gradient_paths(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
    start: Simplex,
    target: Simplex | None = None,
) -> GradientPathsResult:
    """Enumerate the complete bounded gradient-path family from one critical cell."""

    admitted = _admit_acyclic_matching(complex_, pairs)
    matching = admitted.result
    profile = admitted.critical_profile
    critical = set(profile.critical_cells)
    canonical_start = tuple(sorted(start))
    if canonical_start not in critical:
        raise OperationDomainValidationError(
            location=("start",),
            code="topology.discrete_morse.start_not_critical",
            message="the selected gradient-path start must be a critical cell",
        )
    canonical_target: Simplex | None = None
    if target is not None:
        canonical_target = tuple(sorted(target))
        if canonical_target not in critical:
            raise OperationDomainValidationError(
                location=("target",),
                code="topology.discrete_morse.target_not_critical",
                message="the selected gradient-path target must be a critical cell",
            )
        if len(canonical_target) != len(canonical_start) - 1:
            raise OperationDomainValidationError(
                location=("target",),
                code="topology.discrete_morse.target_dimension",
                message="a gradient-path target must lie one dimension below its start",
            )

    cells = _closure_cells(complex_)
    matched, upper = _matching_maps(matching.pairs)
    raw_paths, _states = _enumerate_gradient_paths(
        _faces_by_coface(cells),
        matched,
        upper,
        critical,
        canonical_start,
        canonical_target,
        max_paths=MAX_MORSE_GRADIENT_PATHS,
        max_states=MAX_MORSE_GRADIENT_STATES,
        max_steps=MAX_MORSE_PATH_STEPS,
    )
    ordered = sorted(raw_paths)
    counter = Counter(steps[-1][2] for steps in ordered)
    return GradientPathsResult._from_kernel(
        complex=complex_,
        pairs=matching.pairs,
        critical_profile=profile,
        start=canonical_start,
        target=canonical_target,
        paths=tuple(_gradient_path(steps) for steps in ordered),
        counts_by_target=tuple(
            GradientPathCount(target=terminal, count=count)
            for terminal, count in sorted(counter.items())
        ),
    )


def _morse_betti_numbers(
    basis_cells: tuple[tuple[Simplex, ...], ...],
    boundary_entries: tuple[MorseBoundaryEntry, ...],
) -> tuple[int, ...]:
    """Rank the reduced GF(2) Morse differentials into Betti numbers."""

    ranks: dict[int, int] = {}
    for dimension in range(1, len(basis_cells)):
        columns = basis_cells[dimension]
        rows = basis_cells[dimension - 1]
        column_index = {cell: index for index, cell in enumerate(columns)}
        row_index = {cell: index for index, cell in enumerate(rows)}
        matrix = [[0] * len(columns) for _ in rows]
        for entry in boundary_entries:
            if entry.coefficient and len(entry.source) - 1 == dimension:
                matrix[row_index[entry.target]][column_index[entry.source]] = 1
        ranks[dimension] = prime_field.rank(
            prime_field.PrimeFieldMatrix._from_admitted(
                prime=2,
                entries=tuple(tuple(row) for row in matrix),
                columns=len(columns),
            )
        )
    return tuple(
        len(basis_cells[dimension])
        - ranks.get(dimension, 0)
        - ranks.get(dimension + 1, 0)
        for dimension in range(len(basis_cells))
    )


def compute_morse_complex(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> MorseComplexResult:
    """Compute the graded GF(2) Morse complex of one acyclic matching."""

    admitted = _admit_acyclic_matching(complex_, pairs)
    matching = admitted.result
    profile = admitted.critical_profile
    if len(profile.critical_cells) > MAX_MORSE_CRITICAL_CELLS:
        raise _resource(
            "admission.critical_cells",
            "the matching leaves "
            f"{len(profile.critical_cells)} critical cells, above the "
            f"{MAX_MORSE_CRITICAL_CELLS}-cell Morse-complex envelope",
            ("pairs",),
        )
    critical = set(profile.critical_cells)
    cells = _closure_cells(complex_)
    matched, upper = _matching_maps(matching.pairs)
    faces_of = _faces_by_coface(cells)
    basis_by_dimension = tuple(
        CriticalCellBasis(
            dimension=dimension,
            cells=tuple(
                cell for cell in profile.critical_cells if len(cell) - 1 == dimension
            ),
        )
        for dimension in range(complex_.dimension + 1)
    )

    counts: dict[tuple[Simplex, Simplex], int] = {}
    total_paths = 0
    total_states = 0
    for cell in profile.critical_cells:
        if len(cell) == 1:
            continue
        raw_paths, states = _enumerate_gradient_paths(
            faces_of,
            matched,
            upper,
            critical,
            cell,
            None,
            max_paths=MAX_MORSE_GRADIENT_PATHS - total_paths,
            max_states=MAX_MORSE_GRADIENT_STATES - total_states,
            max_steps=MAX_MORSE_PATH_STEPS,
        )
        total_paths += len(raw_paths)
        total_states += states
        for steps in raw_paths:
            key = (cell, steps[-1][2])
            counts[key] = counts.get(key, 0) + 1
    if len(counts) > MAX_MORSE_BOUNDARY_ENTRIES:
        raise _resource(
            "admission.boundary_entries",
            "the reduced Morse boundary has "
            f"{len(counts)} nonzero gradient-path rows, above the "
            f"{MAX_MORSE_BOUNDARY_ENTRIES}-entry envelope",
            ("pairs",),
        )

    boundary_entries = tuple(
        MorseBoundaryEntry(
            source=source,
            target=terminal,
            gradient_path_count=count,
            coefficient=count % 2,
        )
        for (source, terminal), count in sorted(counts.items())
    )

    rows: dict[Simplex, set[Simplex]] = {}
    for entry in boundary_entries:
        if entry.coefficient:
            rows.setdefault(entry.source, set()).add(entry.target)
    boundary_square_zero = True
    for targets in rows.values():
        accumulated: set[Simplex] = set()
        for terminal in targets:
            accumulated ^= rows.get(terminal, set())
        if accumulated:
            boundary_square_zero = False
            break

    basis_cells = tuple(basis.cells for basis in basis_by_dimension)
    return MorseComplexResult._from_kernel(
        complex=complex_,
        pairs=matching.pairs,
        critical_profile=profile,
        critical_cells_by_dimension=basis_by_dimension,
        boundary_entries=boundary_entries,
        gradient_path_total=total_paths,
        boundary_square_zero=boundary_square_zero,
        morse_euler_characteristic=profile.euler_characteristic,
        closure_euler_characteristic=profile.closure_euler_characteristic,
        betti_numbers=_morse_betti_numbers(basis_cells, boundary_entries),
    )


__all__ = ["compute_gradient_paths", "compute_morse_complex", "construct_matching"]
