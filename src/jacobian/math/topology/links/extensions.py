"""Exact bounded braid and Wirtinger operations for classical links."""

from __future__ import annotations

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
    MAX_WIRTINGER_GENERATORS,
    AlexanderPolynomialResult,
    BraidArtinActionResult,
    BraidClosureResult,
    BraidLetter,
    BraidPermutationResult,
    BraidWord,
    GoeritzCrossingContribution,
    GoeritzDataResult,
    GoeritzRegion,
    LinkDeterminantResult,
    SeifertCircle,
    SeifertCircleResult,
    WirtingerArc,
    WirtingerCrossingRelator,
    WirtingerPresentationResult,
)
from jacobian.math.topology.links._models import (
    ArcPairing,
    LinkCrossing,
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


def _reduce_free_word(letters: Iterable[WordLetter]) -> tuple[WordLetter, ...]:
    reduced: list[WordLetter] = []
    for letter in letters:
        if (
            reduced
            and reduced[-1].generator == letter.generator
            and reduced[-1].exponent == -letter.exponent
        ):
            reduced.pop()
        else:
            reduced.append(letter)
    return tuple(reduced)


def braid_artin_action(word: BraidWord) -> BraidArtinActionResult:
    """Apply the standard Artin action to the free group on braid strands.

    A braid letter acts on the current images from left to right. Its positive
    generator sends ``x_i`` to ``x_i*x_(i+1)*x_i^-1`` and ``x_(i+1)`` to
    ``x_i``; the negative generator uses the inverse automorphism. The output
    is an exact free-group automorphism presentation, not a braid-equivalence
    or word-problem decision procedure.
    """

    admitted = _admit_braid(word)
    images: list[tuple[WordLetter, ...]] = [
        (WordLetter(generator=index, exponent=1),)
        for index in range(admitted.strand_count)
    ]
    total_work = admitted.strand_count
    for letter in admitted.letters:
        i = letter.generator - 1
        first, second = images[i], images[i + 1]
        if letter.exponent == 1:
            inverse_first = tuple(
                WordLetter(generator=item.generator, exponent=-item.exponent)
                for item in reversed(first)
            )
            candidate = _reduce_free_word(first + second + inverse_first)
            replacement = first
            target = i
        else:
            inverse_second = tuple(
                WordLetter(generator=item.generator, exponent=-item.exponent)
                for item in reversed(second)
            )
            candidate = _reduce_free_word(inverse_second + first + second)
            replacement = second
            target = i + 1
        total_work += len(first) + len(second) + len(candidate) + len(replacement)
        if len(candidate) > 128 or total_work > 100_000:
            raise OperationResourceAdmissionError(
                location=("word", "letters"),
                code="link_diagram.artin_action_expansion_bound",
                message=(
                    "the reduced braid action exceeds the 128-letter per-image "
                    "or 100000-letter cumulative substitution envelope"
                ),
            )
        images[target] = candidate
        images[i + 1 if target == i else i] = replacement

    return BraidArtinActionResult(
        word=admitted,
        generator_images=tuple(FiniteGroupWord(letters=image) for image in images),
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
    arcs: list[ArcPairing] = []

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
                arcs.append(ArcPairing(first=previous, second=top))
            last_bottom[position] = bottom_by_position[position]
        crossings.append(
            LinkCrossing(
                crossing_id=crossing_id,
                half_edges=darts,
                over_pair=(0, 2) if letter.exponent == 1 else (1, 3),
                under_pair=(1, 3) if letter.exponent == 1 else (0, 2),
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
            arcs.append(ArcPairing(first=last, second=first))
    arcs.sort(key=lambda arc: (min(arc.first, arc.second), max(arc.first, arc.second)))
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
    strand_partner: dict[str, str] = {}
    arc_partner: dict[str, str] = {}
    for crossing in diagram.crossings:
        for pair in (crossing.over_pair, crossing.under_pair):
            left = crossing.half_edges[pair[0]]
            right = crossing.half_edges[pair[1]]
            strand_partner[left] = right
            strand_partner[right] = left
    for arc in diagram.arcs:
        arc_partner[arc.first] = arc.second
        arc_partner[arc.second] = arc.first

    covered: set[str] = set()
    incoming: set[str] = set()
    for start in sorted(strand_partner):
        if start in covered:
            continue
        current = start
        while current not in covered:
            entry = arc_partner[current]
            exit_dart = strand_partner[entry]
            covered.add(current)
            covered.add(entry)
            incoming.add(entry)
            current = exit_dart
    return incoming


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
        arc_partner[arc.first] = arc.second
        arc_partner[arc.second] = arc.first
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
                "goeritz_face_cycle",
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
        left = face_of[arc.first]
        right = face_of[arc.second]
        if left == right:
            _domain_error(
                ("diagram",),
                "goeritz_checkerboard_loop",
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
                    "goeritz_checkerboard_coloring",
                    "projection regions must admit a checkerboard coloring",
                )
            if neighbor not in colors:
                colors[neighbor] = expected
                queue.append(neighbor)
    if len(colors) != len(adjacency):
        _domain_error(
            ("diagram",),
            "goeritz_connected_projection",
            "this Goeritz slice requires a connected crossing projection",
        )
    return colors


def link_goeritz_data(diagram: OrientedLinkDiagram) -> GoeritzDataResult:
    """Construct a deterministic checkerboard shading and reduced Goeritz matrix."""

    admitted = _admit_diagram(diagram)
    crossing_count = len(admitted.crossings)
    if crossing_count == 0 or admitted.free_loops:
        _domain_error(
            ("diagram",),
            "goeritz_crossing_projection",
            "Goeritz data requires a nonempty crossing projection without free loops",
        )
    if crossing_count > 32:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.goeritz_matrix_bound",
            message="Goeritz matrices are admitted for at most 32 crossings",
        )

    faces, face_of, adjacency = _projection_faces(admitted)
    if len(faces) != crossing_count + 2:
        _domain_error(
            ("diagram",),
            "goeritz_planar_projection",
            "connected crossing projection must satisfy the sphere Euler identity",
        )
    colors = _checkerboard_colors(adjacency)
    regions = tuple(
        GoeritzRegion(
            region_id=f"region_{index:03d}",
            boundary_darts=face,
            shaded=colors[index] == 0,
        )
        for index, face in enumerate(faces)
    )
    shaded_indices = tuple(index for index in range(len(faces)) if colors[index] == 0)
    shaded_positions = {
        region: position for position, region in enumerate(shaded_indices)
    }
    contributions: list[GoeritzCrossingContribution] = []
    matrix = [[0 for _ in shaded_indices] for _ in shaded_indices]
    for crossing in admitted.crossings:
        corner_regions = tuple(face_of[dart] for dart in crossing.half_edges)
        shaded_corners = tuple(
            index for index, region in enumerate(corner_regions) if colors[region] == 0
        )
        if len(shaded_corners) != 2 or (shaded_corners[0] - shaded_corners[1]) % 2:
            _domain_error(
                ("diagram",),
                "goeritz_crossing_shading",
                "checkerboard shading must occupy opposite corners at each crossing",
            )
        first_region = corner_regions[shaded_corners[0]]
        second_region = corner_regions[shaded_corners[1]]
        incidence: Literal[-1, 1] = (
            1 if set(shaded_corners) == set(crossing.over_pair) else -1
        )
        contributions.append(
            GoeritzCrossingContribution(
                crossing_id=crossing.crossing_id,
                first_region_id=regions[first_region].region_id,
                second_region_id=regions[second_region].region_id,
                incidence=incidence,
            )
        )
        first_position = shaded_positions[first_region]
        second_position = shaded_positions[second_region]
        if first_position != second_position:
            matrix[first_position][first_position] += incidence
            matrix[second_position][second_position] += incidence
            matrix[first_position][second_position] -= incidence
            matrix[second_position][first_position] -= incidence
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
        diagram=admitted,
        regions=regions,
        shaded_region_ids=tuple(regions[index].region_id for index in shaded_indices),
        crossing_contributions=tuple(contributions),
        deleted_region_id=regions[shaded_indices[deleted_position]].region_id,
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
        arc_partner[arc.first] = arc.second
        arc_partner[arc.second] = arc.first

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
        union(arc.first, arc.second)
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


__all__ = [
    "braid_closure",
    "braid_inverse",
    "braid_multiply",
    "braid_permutation",
    "link_alexander_polynomial",
    "link_determinant",
    "link_goeritz_data",
    "link_seifert_circles",
    "wirtinger_presentation",
]
