"""Exact bounded braid and Wirtinger operations for classical links."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from math import gcd
from typing import Literal, NoReturn

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.polynomials.values import (
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
)
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)
from jacobian.math.topology.links._extensions_models import (
    MAX_CONWAY_CENTERED_DEGREE,
    MAX_CONWAY_COEFFICIENT_DIGITS,
    MAX_CONWAY_OUTPUT_BYTES,
    MAX_LINK_DISJOINT_UNION_OUTPUT_BYTES,
    MAX_STATE_CIRCLE_CROSSINGS,
    MAX_STATE_CIRCLE_OUTPUT_BYTES,
    MAX_WIRTINGER_GENERATORS,
    AlexanderPolynomialResult,
    BraidClosureResult,
    BraidLetter,
    BraidPermutationResult,
    BraidWord,
    CheckerboardRegion,
    ConwayPolynomialResult,
    GoeritzDataResult,
    LinkBlackboardEdge,
    LinkBlackboardGraph,
    LinkCrossingProfileEntry,
    LinkCrossingProfileResult,
    LinkDeterminantResult,
    LinkDiagramSmoothingState,
    LinkDisjointUnionArcMap,
    LinkDisjointUnionCrossingMap,
    LinkDisjointUnionDartMap,
    LinkDisjointUnionFreeLoopMap,
    LinkDisjointUnionResult,
    LinkSmoothedCircle,
    LinkStateCirclesResult,
    SeifertCircle,
    SeifertCircleResult,
    WirtingerArc,
    WirtingerCrossingRelator,
    WirtingerPresentationResult,
)
from jacobian.math.topology.links._models import (
    MAX_LINK_CROSSINGS,
    LinkCrossing,
    OrientedDiagramArc,
    OrientedLinkDiagram,
)
from jacobian.math.topology.links.operations import link_components


def _domain_error(
    location: tuple[str | int, ...], reason: str, message: str
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"link_diagram.{reason}",
        message=message,
    )


def _admit_braid(value: object) -> BraidWord:
    if not isinstance(value, BraidWord):
        _domain_error(("word",), "braid_word_type", "word must be a BraidWord value")
    try:
        return BraidWord.model_validate_json(value.model_dump_json(warnings=False))
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("word",),
            code="link_diagram.braid_word_shape",
            message="word must satisfy the complete bounded braid-word contract",
        ) from exc


def _admit_diagram(value: object) -> OrientedLinkDiagram:
    if not isinstance(value, OrientedLinkDiagram):
        _domain_error(
            ("diagram",),
            "wirtinger_diagram_type",
            "diagram must be an OrientedLinkDiagram value",
        )
    try:
        return OrientedLinkDiagram.model_validate_json(
            value.model_dump_json(warnings=False)
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("diagram",),
            code="link_diagram.wirtinger_diagram_shape",
            message="diagram must satisfy the complete oriented-link contract",
        ) from exc


def link_disjoint_union(
    diagrams: tuple[OrientedLinkDiagram, ...],
) -> LinkDisjointUnionResult:
    """Form a tagged, source-transporting disjoint union of link diagrams."""

    if not isinstance(diagrams, tuple) or not 1 <= len(diagrams) <= 64:
        _domain_error(
            ("diagrams",),
            "disjoint_union_arity",
            "disjoint union needs between one and 64 diagrams",
        )
    admitted = tuple(_admit_diagram(diagram) for diagram in diagrams)
    crossing_count = sum(len(diagram.crossings) for diagram in admitted)
    free_loop_count = sum(diagram.free_loops for diagram in admitted)
    if crossing_count > MAX_LINK_CROSSINGS:
        raise OperationResourceAdmissionError(
            location=("diagrams",),
            code="link_diagram.disjoint_union_crossing_bound",
            message="the union may contain at most 64 crossings in total",
        )
    if free_loop_count > MAX_LINK_CROSSINGS:
        raise OperationResourceAdmissionError(
            location=("diagrams",),
            code="link_diagram.disjoint_union_free_loop_bound",
            message="the union may contain at most 64 crossing-free components in total",
        )
    source_bytes = sum(
        len(diagram.model_dump_json(warnings=False).encode("utf-8"))
        for diagram in admitted
    )
    if source_bytes * 4 + 16_384 > MAX_LINK_DISJOINT_UNION_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("diagrams",),
            code="link_diagram.disjoint_union_output_bound",
            message="the disjoint union and complete transport exceed the 8 MiB output envelope",
        )

    output_crossings: list[LinkCrossing] = []
    output_arcs: list[OrientedDiagramArc] = []
    crossing_map: list[LinkDisjointUnionCrossingMap] = []
    dart_map: list[LinkDisjointUnionDartMap] = []
    arc_map: list[LinkDisjointUnionArcMap] = []
    free_loop_map: list[LinkDisjointUnionFreeLoopMap] = []
    loop_offset = 0
    for source_index, source in enumerate(admitted):
        source_dart_map: dict[str, str] = {}
        for crossing_index, crossing in enumerate(source.crossings):
            target_crossing_id = (
                f"link_{source_index:02d}_crossing_{crossing_index:03d}"
            )
            target_darts: tuple[str, str, str, str] = (
                f"link_{source_index:02d}_dart_{crossing_index:03d}_0",
                f"link_{source_index:02d}_dart_{crossing_index:03d}_1",
                f"link_{source_index:02d}_dart_{crossing_index:03d}_2",
                f"link_{source_index:02d}_dart_{crossing_index:03d}_3",
            )
            crossing_map.append(
                LinkDisjointUnionCrossingMap(
                    source_index=source_index,
                    source_crossing_id=crossing.crossing_id,
                    target_crossing_id=target_crossing_id,
                )
            )
            for source_dart, target_dart in zip(
                crossing.half_edges, target_darts, strict=True
            ):
                source_dart_map[source_dart] = target_dart
                dart_map.append(
                    LinkDisjointUnionDartMap(
                        source_index=source_index,
                        source_dart_id=source_dart,
                        target_dart_id=target_dart,
                    )
                )
            output_crossings.append(
                LinkCrossing(
                    crossing_id=target_crossing_id,
                    half_edges=target_darts,
                    over_pair=crossing.over_pair,
                    under_pair=crossing.under_pair,
                    sign=crossing.sign,
                )
            )
        for arc in source.arcs:
            target_tail = source_dart_map[arc.tail]
            target_head = source_dart_map[arc.head]
            output_arcs.append(OrientedDiagramArc(tail=target_tail, head=target_head))
            arc_map.append(
                LinkDisjointUnionArcMap(
                    source_index=source_index,
                    source_tail=arc.tail,
                    source_head=arc.head,
                    target_tail=target_tail,
                    target_head=target_head,
                )
            )
        for local_index in range(source.free_loops):
            free_loop_map.append(
                LinkDisjointUnionFreeLoopMap(
                    source_index=source_index,
                    source_loop_index=local_index,
                    target_loop_index=loop_offset + local_index,
                )
            )
        loop_offset += source.free_loops
    output = OrientedLinkDiagram(
        crossings=tuple(output_crossings),
        arcs=tuple(sorted(output_arcs, key=lambda arc: (arc.tail, arc.head))),
        free_loops=loop_offset,
    )
    return LinkDisjointUnionResult(
        sources=admitted,
        diagram=output,
        crossing_map=tuple(crossing_map),
        dart_map=tuple(dart_map),
        arc_map=tuple(arc_map),
        free_loop_map=tuple(free_loop_map),
    )


def _cycles(permutation: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in range(len(permutation)):
        if start in visited:
            continue
        cycle: list[int] = []
        cursor = start
        while cursor not in visited:
            visited.add(cursor)
            cycle.append(cursor)
            cursor = permutation[cursor]
        least = min(range(len(cycle)), key=cycle.__getitem__)
        cycles.append(tuple(cycle[least:] + cycle[:least]))
    return tuple(sorted(cycles))


def braid_permutation(word: BraidWord) -> BraidPermutationResult:
    """Return the exact strand permutation and closure cycles of a braid word."""

    admitted = _admit_braid(word)
    positions = list(range(admitted.strand_count))
    for letter in admitted.letters:
        left = letter.generator - 1
        positions[left], positions[left + 1] = positions[left + 1], positions[left]
    permutation_rows = [0] * admitted.strand_count
    for final_position, source_strand in enumerate(positions):
        permutation_rows[source_strand] = final_position
    permutation = tuple(permutation_rows)
    cycles = _cycles(permutation)
    return BraidPermutationResult(
        word=admitted,
        permutation=permutation,
        cycles=cycles,
        closure_component_count=len(cycles),
        exponent_sum=sum(letter.exponent for letter in admitted.letters),
    )


def braid_multiply(left: BraidWord, right: BraidWord) -> BraidWord:
    """Concatenate two presentation words in the same braid group."""

    admitted_left = _admit_braid(left)
    admitted_right = _admit_braid(right)
    if admitted_left.strand_count != admitted_right.strand_count:
        _domain_error(
            ("right",),
            "braid_parent_mismatch",
            "braid multiplication requires the same strand count",
        )
    if len(admitted_left.letters) + len(admitted_right.letters) > 64:
        raise OperationResourceAdmissionError(
            location=("right",),
            code="link_diagram.braid_product_length_bound",
            message="concatenated braid word exceeds the 64-letter envelope",
        )
    return BraidWord(
        strand_count=admitted_left.strand_count,
        letters=admitted_left.letters + admitted_right.letters,
    )


def braid_inverse(word: BraidWord) -> BraidWord:
    """Reverse a braid word and invert every Artin generator."""

    admitted = _admit_braid(word)
    return BraidWord(
        strand_count=admitted.strand_count,
        letters=tuple(
            BraidLetter(generator=letter.generator, exponent=-letter.exponent)
            for letter in reversed(admitted.letters)
        ),
    )


def braid_closure(word: BraidWord) -> BraidClosureResult:
    """Construct the standard oriented closure of a bounded braid word."""

    admitted = _admit_braid(word)
    first_top: list[str | None] = [None] * admitted.strand_count
    last_bottom: list[str | None] = [None] * admitted.strand_count
    crossings: list[LinkCrossing] = []
    arcs: list[OrientedDiagramArc] = []

    for crossing_index, letter in enumerate(admitted.letters):
        crossing_id = f"crossing_{crossing_index:03d}"
        darts = (
            f"{crossing_id}:dart_0",
            f"{crossing_id}:dart_1",
            f"{crossing_id}:dart_2",
            f"{crossing_id}:dart_3",
        )
        left = letter.generator - 1
        right = left + 1
        top_by_position = {left: darts[0], right: darts[1]}
        bottom_by_position = {left: darts[3], right: darts[2]}
        for position in (left, right):
            top = top_by_position[position]
            previous = last_bottom[position]
            if previous is None:
                first_top[position] = top
            else:
                arcs.append(OrientedDiagramArc(tail=previous, head=top))
            last_bottom[position] = bottom_by_position[position]
        ccw_darts = (darts[0], darts[3], darts[2], darts[1])
        old_to_new = {0: 0, 3: 1, 2: 2, 1: 3}
        old_over = (0, 2) if letter.exponent == 1 else (1, 3)
        mapped_over = tuple(old_to_new[index] for index in old_over)
        over_pair = (min(mapped_over), max(mapped_over))
        under_pair = (1, 3) if over_pair == (0, 2) else (0, 2)
        crossings.append(
            LinkCrossing(
                crossing_id=crossing_id,
                half_edges=ccw_darts,
                over_pair=over_pair,
                under_pair=under_pair,
                sign=letter.exponent,
            )
        )

    free_loops = 0
    for position in range(admitted.strand_count):
        first = first_top[position]
        last = last_bottom[position]
        if first is None or last is None:
            free_loops += 1
        else:
            arcs.append(OrientedDiagramArc(tail=last, head=first))
    arcs.sort(key=lambda arc: (arc.tail, arc.head))
    diagram = OrientedLinkDiagram(
        crossings=tuple(crossings),
        arcs=tuple(arcs),
        free_loops=free_loops,
    )
    return BraidClosureResult(
        word=admitted,
        permutation=braid_permutation(admitted),
        diagram=diagram,
    )


def _union_find(
    labels: Iterable[str],
) -> tuple[Callable[[str], str], Callable[[str, str], None]]:
    parent = {label: label for label in labels}

    def find(label: str) -> str:
        cursor = label
        while parent[cursor] != cursor:
            parent[cursor] = parent[parent[cursor]]
            cursor = parent[cursor]
        return cursor

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    return find, union


def _reduced_word(
    letters: Iterable[tuple[int, Literal[-1, 1]]],
) -> FiniteGroupWord:
    stack: list[tuple[int, Literal[-1, 1]]] = []
    for generator, exponent in letters:
        if stack and stack[-1] == (generator, -exponent):
            stack.pop()
        else:
            stack.append((generator, exponent))
    return FiniteGroupWord(
        letters=tuple(
            WordLetter(generator=generator, exponent=exponent)
            for generator, exponent in stack
        )
    )


def _incoming_darts(diagram: OrientedLinkDiagram) -> set[str]:
    """Return source darts where the encoded orientation enters crossings."""
    return {arc.head for arc in diagram.arcs}


def _integer_determinant(matrix: tuple[tuple[int, ...], ...]) -> int:
    size = len(matrix)
    if size == 0:
        return 1
    work = [list(row) for row in matrix]
    sign = 1
    previous_pivot = 1
    for pivot_index in range(size - 1):
        if work[pivot_index][pivot_index] == 0:
            replacement = next(
                (
                    row
                    for row in range(pivot_index + 1, size)
                    if work[row][pivot_index] != 0
                ),
                None,
            )
            if replacement is None:
                return 0
            work[pivot_index], work[replacement] = (
                work[replacement],
                work[pivot_index],
            )
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                if numerator % previous_pivot:
                    raise RuntimeError("Bareiss division was unexpectedly inexact")
                work[row][column] = numerator // previous_pivot
        previous_pivot = pivot
    return sign * work[-1][-1]


def _projection_faces(
    diagram: OrientedLinkDiagram,
) -> tuple[tuple[tuple[str, ...], ...], dict[str, int], dict[int, set[int]]]:
    cyclic_successor: dict[str, str] = {}
    arc_partner: dict[str, str] = {}
    for crossing in diagram.crossings:
        for index, dart in enumerate(crossing.half_edges):
            cyclic_successor[dart] = crossing.half_edges[(index + 1) % 4]
    for arc in diagram.arcs:
        arc_partner[arc.tail] = arc.head
        arc_partner[arc.head] = arc.tail
    face_successor = {
        dart: cyclic_successor[arc_partner[dart]] for dart in cyclic_successor
    }
    raw_faces: list[tuple[str, ...]] = []
    covered: set[str] = set()
    for start in sorted(face_successor):
        if start in covered:
            continue
        cycle: list[str] = []
        current = start
        while current not in covered:
            covered.add(current)
            cycle.append(current)
            current = face_successor[current]
        if current != start:
            _domain_error(
                ("diagram",),
                "blackboard_graph_face_cycle",
                "rotation and arc permutations must close into face cycles",
            )
        least = min(range(len(cycle)), key=cycle.__getitem__)
        raw_faces.append(tuple(cycle[least:] + cycle[:least]))
    faces = tuple(sorted(raw_faces))
    face_of = {
        dart: face_index for face_index, face in enumerate(faces) for dart in face
    }
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(faces))}
    for arc in diagram.arcs:
        left = face_of[arc.tail]
        right = face_of[arc.head]
        if left == right:
            _domain_error(
                ("diagram",),
                "blackboard_graph_region_adjacency",
                "every projection edge must separate two checkerboard regions",
            )
        adjacency[left].add(right)
        adjacency[right].add(left)
    return faces, face_of, adjacency


def _checkerboard_colors(adjacency: dict[int, set[int]]) -> dict[int, int]:
    colors = {0: 0}
    queue = [0]
    while queue:
        region = queue.pop()
        for neighbor in adjacency[region]:
            expected = 1 - colors[region]
            if neighbor in colors and colors[neighbor] != expected:
                _domain_error(
                    ("diagram",),
                    "blackboard_graph_coloring",
                    "projection regions must admit a checkerboard coloring",
                )
            if neighbor not in colors:
                colors[neighbor] = expected
                queue.append(neighbor)
    if len(colors) != len(adjacency):
        _domain_error(
            ("diagram",),
            "blackboard_graph_connected_projection",
            "this Goeritz slice requires a connected crossing projection",
        )
    return colors


def _construct_blackboard_graph(
    admitted: OrientedLinkDiagram,
) -> LinkBlackboardGraph:
    """Construct a Tait graph from one already-admitted source diagram."""
    crossing_count = len(admitted.crossings)
    if crossing_count == 0 or admitted.free_loops:
        _domain_error(
            ("diagram",),
            "blackboard_graph_crossing_projection",
            "Goeritz data requires a nonempty crossing projection without free loops",
        )
    if crossing_count > MAX_LINK_CROSSINGS:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.blackboard_graph_bound",
            message="blackboard graphs are admitted for at most 64 crossings",
        )
    # A returned value repeats each bounded source label in crossing/arc rows,
    # face cycles, and crossing-edge transport. This conservative bound is
    # checked before computing any face permutation or result objects.
    output_bound = 8192 + crossing_count * 6000 + (crossing_count + 2) * 1024
    if output_bound > 512 * 1024:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.blackboard_graph_output_bound",
            message="checkerboard graph result exceeds its 512 KiB output envelope",
        )

    faces, face_of, adjacency = _projection_faces(admitted)
    if len(faces) != crossing_count + 2:
        _domain_error(
            ("diagram",),
            "blackboard_graph_planar_projection",
            "connected crossing projection must satisfy the sphere Euler identity",
        )
    colors = _checkerboard_colors(adjacency)
    regions = tuple(
        CheckerboardRegion(
            region_id=f"region_{index:03d}",
            boundary_darts=face,
            shaded=colors[index] == 0,
        )
        for index, face in enumerate(faces)
    )
    shaded_indices = tuple(index for index in range(len(faces)) if colors[index] == 0)
    edges: list[LinkBlackboardEdge] = []
    for crossing in admitted.crossings:
        corner_regions = tuple(face_of[dart] for dart in crossing.half_edges)
        shaded_corners = tuple(
            index for index, region in enumerate(corner_regions) if colors[region] == 0
        )
        if len(shaded_corners) != 2 or (shaded_corners[0] - shaded_corners[1]) % 2:
            _domain_error(
                ("diagram",),
                "blackboard_graph_crossing_shading",
                "checkerboard shading must occupy opposite corners at each crossing",
            )
        first_region = corner_regions[shaded_corners[0]]
        second_region = corner_regions[shaded_corners[1]]
        tait_sign: Literal[-1, 1] = (
            1 if set(shaded_corners) == set(crossing.over_pair) else -1
        )
        edges.append(
            LinkBlackboardEdge(
                crossing_id=crossing.crossing_id,
                first_region_id=regions[first_region].region_id,
                second_region_id=regions[second_region].region_id,
                first_corner_index=shaded_corners[0],
                second_corner_index=shaded_corners[1],
                tait_sign=tait_sign,
            )
        )
    return LinkBlackboardGraph(
        diagram=admitted,
        regions=regions,
        shaded_region_ids=tuple(regions[index].region_id for index in shaded_indices),
        edges=tuple(edges),
    )


def link_blackboard_graph(diagram: OrientedLinkDiagram) -> LinkBlackboardGraph:
    """Construct the canonical signed Tait graph with source region transport."""
    return _construct_blackboard_graph(_admit_diagram(diagram))


def link_goeritz_data(diagram: OrientedLinkDiagram) -> GoeritzDataResult:
    """Construct the reduced Goeritz matrix from the canonical Tait graph."""
    admitted = _admit_diagram(diagram)
    if len(admitted.crossings) > 32:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.goeritz_matrix_bound",
            message="Goeritz matrices are admitted for at most 32 crossings",
        )
    graph = _construct_blackboard_graph(admitted)
    shaded_indices = graph.shaded_region_ids
    shaded_positions = {
        region: position for position, region in enumerate(shaded_indices)
    }
    matrix = [[0 for _ in shaded_indices] for _ in shaded_indices]
    for edge in graph.edges:
        first_position = shaded_positions[edge.first_region_id]
        second_position = shaded_positions[edge.second_region_id]
        if first_position != second_position:
            matrix[first_position][first_position] += edge.tait_sign
            matrix[second_position][second_position] += edge.tait_sign
            matrix[first_position][second_position] -= edge.tait_sign
            matrix[second_position][first_position] -= edge.tait_sign
    deleted_position = len(shaded_indices) - 1
    reduced_entries = tuple(
        tuple(value for column, value in enumerate(row) if column != deleted_position)
        for row_index, row in enumerate(matrix)
        if row_index != deleted_position
    )
    reduced = IntegerMatrix(
        entries=reduced_entries,
        row_count=len(reduced_entries),
        column_count=len(reduced_entries),
    )
    return GoeritzDataResult(
        blackboard_graph=graph,
        deleted_region_id=shaded_indices[deleted_position],
        reduced_matrix=reduced,
        absolute_determinant=abs(_integer_determinant(reduced_entries)),
    )


def link_seifert_circles(diagram: OrientedLinkDiagram) -> SeifertCircleResult:
    """Apply every oriented smoothing and return the canonical Seifert circles."""

    admitted = _admit_diagram(diagram)
    components = link_components(admitted)
    if len(components.components) != 1:
        _domain_error(
            ("diagram",),
            "seifert_requires_knot",
            "this canonical-orientation Seifert contract requires one component",
        )
    if not admitted.crossings:
        return SeifertCircleResult(
            diagram=admitted,
            circles=(SeifertCircle(circle_id="seifert_circle_000", darts=()),),
            band_crossing_ids=(),
            euler_characteristic=1,
            genus=0,
        )

    incoming = _incoming_darts(admitted)
    smoothing_partner: dict[str, str] = {}
    for crossing in admitted.crossings:
        darts = crossing.half_edges
        candidates = (
            ((darts[0], darts[1]), (darts[2], darts[3])),
            ((darts[1], darts[2]), (darts[3], darts[0])),
        )
        valid = tuple(
            pairing
            for pairing in candidates
            if all((left in incoming) != (right in incoming) for left, right in pairing)
        )
        if len(valid) != 1:
            _domain_error(
                ("diagram",),
                "seifert_orientation_incidence",
                "oriented crossing incidence must determine one planar smoothing",
            )
        for left, right in valid[0]:
            smoothing_partner[left] = right
            smoothing_partner[right] = left
    arc_partner: dict[str, str] = {}
    for arc in admitted.arcs:
        arc_partner[arc.tail] = arc.head
        arc_partner[arc.head] = arc.tail

    covered: set[str] = set()
    circle_rows: list[SeifertCircle] = []
    for start in sorted(smoothing_partner):
        if start in covered:
            continue
        cycle: list[str] = []
        current = start
        while current not in covered:
            entry = arc_partner[current]
            covered.add(current)
            covered.add(entry)
            cycle.extend((current, entry))
            current = smoothing_partner[entry]
        if current != start:
            _domain_error(
                ("diagram",),
                "seifert_smoothing_cycle",
                "oriented smoothing must close into disjoint circles",
            )
        circle_rows.append(
            SeifertCircle(
                circle_id=f"seifert_circle_{len(circle_rows):03d}",
                darts=tuple(cycle),
            )
        )
    euler_characteristic = len(circle_rows) - len(admitted.crossings)
    genus_numerator = 1 - euler_characteristic
    if genus_numerator < 0 or genus_numerator % 2:
        _domain_error(
            ("diagram",),
            "seifert_surface_genus",
            "Seifert disk-band surface must have a nonnegative integral genus",
        )
    return SeifertCircleResult(
        diagram=admitted,
        circles=tuple(circle_rows),
        band_crossing_ids=tuple(
            crossing.crossing_id for crossing in admitted.crossings
        ),
        euler_characteristic=euler_characteristic,
        genus=genus_numerator // 2,
    )


def wirtinger_presentation(
    diagram: OrientedLinkDiagram,
) -> WirtingerPresentationResult:
    """Construct one exact finite Wirtinger presentation of a link diagram."""

    admitted = _admit_diagram(diagram)
    darts = tuple(
        dart for crossing in admitted.crossings for dart in crossing.half_edges
    )
    find, union = _union_find(darts)
    for arc in admitted.arcs:
        union(arc.tail, arc.head)
    for crossing in admitted.crossings:
        over_left = crossing.half_edges[crossing.over_pair[0]]
        over_right = crossing.half_edges[crossing.over_pair[1]]
        union(over_left, over_right)

    classes: dict[str, list[str]] = {}
    for dart in darts:
        classes.setdefault(find(dart), []).append(dart)
    ordered_classes = sorted(
        (tuple(sorted(members)) for members in classes.values()),
        key=lambda members: members[0],
    )
    generator_count = len(ordered_classes) + admitted.free_loops
    if generator_count > MAX_WIRTINGER_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.wirtinger_generator_bound",
            message=(
                "Wirtinger arc count exceeds the 64-generator finite-presentation "
                "envelope"
            ),
        )
    arcs = [
        WirtingerArc(generator_id=f"meridian_{index:03d}", darts=members)
        for index, members in enumerate(ordered_classes)
    ]
    first_free_generator = len(arcs)
    arcs.extend(
        WirtingerArc(
            generator_id=f"meridian_{first_free_generator + loop_index:03d}",
            darts=(),
        )
        for loop_index in range(admitted.free_loops)
    )
    dart_generator = {
        dart: index for index, members in enumerate(ordered_classes) for dart in members
    }
    incoming = _incoming_darts(admitted)
    relator_rows: list[WirtingerCrossingRelator] = []
    for crossing in admitted.crossings:
        over = dart_generator[crossing.half_edges[crossing.over_pair[0]]]
        under_darts = tuple(crossing.half_edges[index] for index in crossing.under_pair)
        incoming_under = next(dart for dart in under_darts if dart in incoming)
        outgoing_under = next(dart for dart in under_darts if dart != incoming_under)
        under_in = dart_generator[incoming_under]
        under_out = dart_generator[outgoing_under]
        letters: tuple[tuple[int, Literal[-1, 1]], ...] = (
            ((over, 1), (under_in, 1), (over, -1), (under_out, -1))
            if crossing.sign == 1
            else ((over, -1), (under_in, 1), (over, 1), (under_out, -1))
        )
        word = _reduced_word(letters)
        relator_rows.append(
            WirtingerCrossingRelator(
                crossing_id=crossing.crossing_id,
                over_generator=over,
                under_incoming_generator=under_in,
                under_outgoing_generator=under_out,
                word=word,
            )
        )
    presentation = FiniteGroupPresentation(
        generators=tuple(arc.generator_id for arc in arcs),
        relators=tuple(row.word for row in relator_rows),
    )
    return WirtingerPresentationResult(
        diagram=admitted,
        arcs=tuple(arcs),
        crossing_relators=tuple(relator_rows),
        presentation=presentation,
    )


_IntegerLaurent = dict[int, int]


def _laurent_add(
    target: _IntegerLaurent, source: _IntegerLaurent, sign: int = 1
) -> None:
    for exponent, coefficient in source.items():
        value = target.get(exponent, 0) + sign * coefficient
        if value:
            target[exponent] = value
        else:
            target.pop(exponent, None)


def _laurent_multiply(left: _IntegerLaurent, right: _IntegerLaurent) -> _IntegerLaurent:
    result: _IntegerLaurent = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = left_exponent + right_exponent
            result[exponent] = result.get(exponent, 0) + (
                left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _laurent_determinant(
    matrix: tuple[tuple[_IntegerLaurent, ...], ...],
) -> _IntegerLaurent:
    size = len(matrix)
    if size == 0:
        return {0: 1}
    rows: dict[int, _IntegerLaurent] = {0: {0: 1}}
    for row_index in range(size):
        next_rows: dict[int, _IntegerLaurent] = {}
        for mask, coefficient in rows.items():
            for column in range(size):
                if mask & (1 << column):
                    continue
                product = _laurent_multiply(coefficient, matrix[row_index][column])
                if not product:
                    continue
                inversions = sum(
                    1 for used in range(size) if mask & (1 << used) and used > column
                )
                destination = next_rows.setdefault(mask | (1 << column), {})
                _laurent_add(destination, product, -1 if inversions % 2 else 1)
        rows = next_rows
    return rows.get((1 << size) - 1, {})


def _fox_alexander_matrix(
    presentation: FiniteGroupPresentation,
) -> tuple[tuple[_IntegerLaurent, ...], ...]:
    generator_count = len(presentation.generators)
    matrix: list[tuple[_IntegerLaurent, ...]] = []
    for relator in presentation.relators:
        row: list[_IntegerLaurent] = [{} for _ in range(generator_count)]
        prefix_exponent = 0
        for letter in relator.letters:
            if letter.exponent == 1:
                contribution_exponent = prefix_exponent
                prefix_exponent += 1
                contribution = 1
            else:
                prefix_exponent -= 1
                contribution_exponent = prefix_exponent
                contribution = -1
            column = row[letter.generator]
            column[contribution_exponent] = (
                column.get(contribution_exponent, 0) + contribution
            )
            if column[contribution_exponent] == 0:
                del column[contribution_exponent]
        matrix.append(tuple(row))
    return tuple(matrix)


def _normalized_alexander(terms: _IntegerLaurent) -> RationalLaurentPolynomial:
    if not terms:
        return RationalLaurentPolynomial(variables=("t",), terms=())
    divisor = 0
    for coefficient in terms.values():
        divisor = gcd(divisor, abs(coefficient))
    least_exponent = min(terms)
    normalized = {
        exponent - least_exponent: coefficient // divisor
        for exponent, coefficient in terms.items()
    }
    if normalized[0] < 0:
        normalized = {
            exponent: -coefficient for exponent, coefficient in normalized.items()
        }
    return RationalLaurentPolynomial(
        variables=("t",),
        terms=tuple(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=coefficient, den=1),
                exponents=(exponent,),
            )
            for exponent, coefficient in sorted(normalized.items(), reverse=True)
        ),
    )


def link_alexander_polynomial(
    diagram: OrientedLinkDiagram,
) -> AlexanderPolynomialResult:
    """Compute the normalized one-variable Alexander polynomial of a knot."""

    admitted = _admit_diagram(diagram)
    components = link_components(admitted)
    if len(components.components) != 1:
        _domain_error(
            ("diagram",),
            "alexander_requires_knot",
            "this one-variable Alexander contract requires exactly one component",
        )
    crossing_count = len(admitted.crossings)
    if crossing_count > 8:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.alexander_crossing_bound",
            message=(
                "Alexander determinant expansion is admitted for at most eight "
                "crossings"
            ),
        )
    if crossing_count == 0:
        polynomial = _normalized_alexander({0: 1})
    else:
        wirtinger = wirtinger_presentation(admitted)
        generator_count = len(wirtinger.presentation.generators)
        if generator_count != crossing_count:
            _domain_error(
                ("diagram",),
                "alexander_wirtinger_deficiency",
                "knot diagram must yield one Wirtinger generator per crossing",
            )
        matrix = _fox_alexander_matrix(wirtinger.presentation)
        minor = tuple(tuple(row[:-1]) for row in matrix[:-1])
        determinant_terms = _laurent_determinant(minor)
        if not determinant_terms:
            raise RuntimeError(
                "the admitted knot Wirtinger minor unexpectedly vanished"
            )
        polynomial = _normalized_alexander(determinant_terms)
    return AlexanderPolynomialResult(diagram=admitted, polynomial=polynomial)


def link_conway_polynomial(
    diagram: OrientedLinkDiagram,
) -> ConwayPolynomialResult:
    """Return the normalized knot Conway polynomial from the exact Alexander value."""
    return _conway_from_alexander(link_alexander_polynomial(diagram))


def _conway_from_alexander(
    alexander: AlexanderPolynomialResult,
) -> ConwayPolynomialResult:
    """Return the normalized knot Conway polynomial from the exact Alexander value.

    For knots the Alexander polynomial is symmetric after a Laurent shift.
    With ``x = t + t^-1 = z^2 + 2``, each symmetric pair
    ``t^k + t^-k`` is converted by an exact integer recurrence. The normalization
    ``Delta(1)=1`` fixes the sign and gives ``nabla(0)=1``.
    """
    centered_terms, degree = _center_normalized_alexander(alexander)
    _admit_conway_expansion(alexander, centered_terms, degree)
    conway_terms = _expand_conway_coefficients(centered_terms, degree)
    polynomial = RationalLaurentPolynomial(
        variables=("z",),
        terms=tuple(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=coefficient, den=1),
                exponents=(exponent,),
            )
            for exponent, coefficient in sorted(conway_terms.items(), reverse=True)
        ),
    )
    return ConwayPolynomialResult(alexander=alexander, polynomial=polynomial)


def _center_normalized_alexander(
    alexander: AlexanderPolynomialResult,
) -> tuple[dict[int, int], int]:
    source_terms: dict[int, int] = {}
    for term in alexander.polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        if coefficient.denominator != 1:
            raise RuntimeError("knot Alexander coefficients must be integral")
        source_terms[term.exponents[0]] = coefficient.numerator
    if not source_terms:
        raise RuntimeError("a knot Alexander polynomial cannot be zero")

    least_exponent = min(source_terms)
    greatest_exponent = max(source_terms)
    exponent_span = greatest_exponent - least_exponent
    if exponent_span % 2:
        raise RuntimeError("knot Alexander support must have an integral center")
    center = (least_exponent + greatest_exponent) // 2
    centered_terms = {
        exponent - center: coefficient for exponent, coefficient in source_terms.items()
    }
    degree = max(abs(exponent) for exponent in centered_terms)
    if degree > MAX_CONWAY_CENTERED_DEGREE:
        raise OperationResourceAdmissionError(
            location=("diagram", "alexander", "polynomial"),
            code="link_diagram.conway_degree_bound",
            message="Conway conversion is bounded to centered Alexander degree 64",
        )
    if any(
        centered_terms.get(exponent, 0) != centered_terms.get(-exponent, 0)
        for exponent in range(1, degree + 1)
    ):
        raise RuntimeError("knot Alexander coefficients must be reciprocal")
    augmentation = sum(centered_terms.values())
    if augmentation not in (-1, 1):
        raise RuntimeError(
            "knot Alexander polynomial must evaluate to plus or minus one at 1"
        )
    if augmentation == -1:
        centered_terms = {
            exponent: -coefficient for exponent, coefficient in centered_terms.items()
        }
    return centered_terms, degree


def _admit_conway_expansion(
    alexander: AlexanderPolynomialResult,
    centered_terms: dict[int, int],
    degree: int,
) -> None:
    maximum_input_bits = max(
        abs(value).bit_length() for value in centered_terms.values()
    )
    maximum_input_digits = (maximum_input_bits * 30_103 + 99_999) // 100_000
    coefficient_digits_bound = (
        maximum_input_digits + 2 * (degree + 1) + len(centered_terms)
    )
    work_bound = 4 * (degree + 1) ** 2
    alexander_bytes = len(alexander.model_dump_json(warnings=False).encode())
    output_bytes_bound = (
        alexander_bytes + 256 + (degree + 1) * (2 * coefficient_digits_bound + 128)
    )
    if (
        coefficient_digits_bound > MAX_CONWAY_COEFFICIENT_DIGITS
        or work_bound > 100_000
        or output_bytes_bound > MAX_CONWAY_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("diagram", "alexander", "polynomial"),
            code="link_diagram.conway_output_bound",
            message="exact Conway conversion exceeds its coefficient, work, or output bound",
        )


def _expand_conway_coefficients(
    centered_terms: dict[int, int], degree: int
) -> dict[int, int]:
    conway_terms: dict[int, int] = {0: centered_terms.get(0, 0)}
    if degree:
        previous_previous = {0: 2}
        previous = {0: 2, 2: 1}
        for exponent, coefficient in previous.items():
            conway_terms[exponent] = (
                conway_terms.get(exponent, 0) + centered_terms.get(1, 0) * coefficient
            )
        for index in range(2, degree + 1):
            current: dict[int, int] = {}
            for exponent, coefficient in previous.items():
                current[exponent] = current.get(exponent, 0) + 2 * coefficient
                current[exponent + 2] = current.get(exponent + 2, 0) + coefficient
            for exponent, coefficient in previous_previous.items():
                current[exponent] = current.get(exponent, 0) - coefficient
            current = {exponent: value for exponent, value in current.items() if value}
            scalar = centered_terms.get(index, 0)
            if scalar:
                for exponent, coefficient in current.items():
                    conway_terms[exponent] = (
                        conway_terms.get(exponent, 0) + scalar * coefficient
                    )
            previous_previous, previous = previous, current
    conway_terms = {
        exponent: coefficient
        for exponent, coefficient in conway_terms.items()
        if coefficient
    }
    return conway_terms


def link_determinant(diagram: OrientedLinkDiagram) -> LinkDeterminantResult:
    """Return ``abs(Delta_K(-1))`` under the knot Alexander convention."""

    alexander = link_alexander_polynomial(diagram)
    evaluation = sum(
        (term.coefficient.as_fraction() * ((-1) ** term.exponents[0]))
        for term in alexander.polynomial.terms
    )
    if evaluation.denominator != 1:
        raise RuntimeError("the normalized Alexander evaluation must be integral")
    integer_evaluation = evaluation.numerator
    return LinkDeterminantResult(
        alexander=alexander,
        evaluation_at_minus_one=integer_evaluation,
        determinant=abs(integer_evaluation),
    )


def link_crossing_profile(
    diagram: OrientedLinkDiagram,
) -> LinkCrossingProfileResult:
    """Return each crossing's sign and ordered over/under component pair."""
    components = link_components(diagram)
    component_by_role: dict[tuple[str, str], set[str]] = {}
    for component in components.components:
        for visit in component.visits:
            component_by_role.setdefault((visit.crossing_id, visit.role), set()).add(
                component.component_id
            )
    entries = []
    for crossing in components.diagram.crossings:
        over_ids = component_by_role.get((crossing.crossing_id, "OVER"), set())
        under_ids = component_by_role.get((crossing.crossing_id, "UNDER"), set())
        if len(over_ids) != 1 or len(under_ids) != 1:
            raise OperationDomainValidationError(
                location=("diagram", "crossings", crossing.crossing_id),
                code="link_diagram.crossing_profile_component_roles",
                message="each crossing strand must belong to exactly one diagram component",
            )
        entries.append(
            LinkCrossingProfileEntry(
                crossing_id=crossing.crossing_id,
                sign=crossing.sign,
                over_component_id=next(iter(over_ids)),
                under_component_id=next(iter(under_ids)),
            )
        )
    return LinkCrossingProfileResult(
        components=components,
        crossings=tuple(entries),
        writhe=sum(entry.sign for entry in entries),
    )


def link_state_circles(
    state: LinkDiagramSmoothingState,
) -> LinkStateCirclesResult:
    """Return the exact cyclic dart circles of one complete A/B state."""
    if not isinstance(state, LinkDiagramSmoothingState):
        _domain_error(
            ("state",),
            "smoothing_state_type",
            "state must be a complete LinkDiagramSmoothingState value",
        )
    try:
        admitted = LinkDiagramSmoothingState.model_validate_json(
            state.model_dump_json(warnings=False)
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("state",),
            code="link_diagram.smoothing_state_shape",
            message="state must satisfy the complete source crossing-axis contract",
        ) from exc
    diagram = admitted.diagram
    crossing_count = len(diagram.crossings)
    if crossing_count > MAX_STATE_CIRCLE_CROSSINGS:
        raise OperationResourceAdmissionError(
            location=("state", "diagram", "crossings"),
            code="link_diagram.state_circles_state_bound",
            message="state-circle computation is bounded to 64 crossings",
        )
    dart_labels = tuple(
        dart for crossing in diagram.crossings for dart in crossing.half_edges
    )
    label_bytes = sum(
        len(json.dumps(dart, ensure_ascii=True, separators=(",", ":")).encode())
        for dart in dart_labels
    )
    # The one requested state repeats every dart exactly once, with JSON
    # delimiters for the circle partition and crossing-choice ledger. Escaped
    # label lengths account for Unicode scalar labels.
    output_bound = (
        len(diagram.model_dump_json(warnings=False).encode())
        + label_bytes
        + (20 * len(dart_labels) + 8 * crossing_count + 16 * diagram.free_loops + 128)
    )
    if output_bound > MAX_STATE_CIRCLE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("state",),
            code="link_diagram.state_circles_output_bound",
            message="the state-circle result exceeds the 8 MiB output bound",
        )
    if 2 * len(dart_labels) + crossing_count > 4_096:
        raise OperationResourceAdmissionError(
            location=("state",),
            code="link_diagram.state_circles_work_bound",
            message="the state-circle traversal exceeds its work bound",
        )

    if not dart_labels:
        circles = tuple(LinkSmoothedCircle(darts=()) for _ in range(diagram.free_loops))
    else:
        arc_mate: dict[str, str] = {}
        for arc in diagram.arcs:
            arc_mate[arc.tail] = arc.head
            arc_mate[arc.head] = arc.tail
        smoothing_mate: dict[str, str] = {}
        for crossing, choice in zip(diagram.crossings, admitted.choices, strict=True):
            over_even = set(crossing.over_pair) == {0, 2}
            smoothing_index = (
                (0 if choice == "A" else 1)
                if over_even
                else (1 if choice == "A" else 0)
            )
            pairs = ((0, 1, 2, 3), (1, 2, 3, 0))[smoothing_index]
            half_edges = crossing.half_edges
            left, right = half_edges[pairs[0]], half_edges[pairs[1]]
            smoothing_mate[left] = right
            smoothing_mate[right] = left
            left, right = half_edges[pairs[2]], half_edges[pairs[3]]
            smoothing_mate[left] = right
            smoothing_mate[right] = left

        unseen = set(dart_labels)
        circle_rows: list[tuple[str, ...]] = []
        while unseen:
            start = min(unseen)
            candidates: list[tuple[str, ...]] = []
            for first_edge in ("arc", "smooth"):
                row: list[str] = []
                current = start
                edge = first_edge
                while True:
                    row.append(current)
                    current = (
                        arc_mate[current] if edge == "arc" else smoothing_mate[current]
                    )
                    edge = "smooth" if edge == "arc" else "arc"
                    if current == start and edge == first_edge:
                        break
                    if current in row:
                        _domain_error(
                            ("state",),
                            "state_circles_not_cycles",
                            "smoothing pairings must form disjoint dart cycles",
                        )
                candidates.append(tuple(row))
            cycle = min(candidates)
            circle_rows.append(cycle)
            unseen.difference_update(cycle)
        circles = tuple(LinkSmoothedCircle(darts=row) for row in sorted(circle_rows))
    return LinkStateCirclesResult(
        state=admitted,
        circles=circles,
        circle_count=len(circles),
    )


__all__ = [
    "braid_closure",
    "braid_inverse",
    "braid_multiply",
    "braid_permutation",
    "link_alexander_polynomial",
    "link_conway_polynomial",
    "link_crossing_profile",
    "link_determinant",
    "link_goeritz_data",
    "link_seifert_circles",
    "link_state_circles",
    "wirtinger_presentation",
]
