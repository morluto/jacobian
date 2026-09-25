"""Native exact link-diagram component traversal."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import product
from typing import NoReturn

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import RationalLaurentPolynomial
from jacobian.math.topology.links._models import (
    MAX_LINK_BRACKET_CROSSINGS,
    MAX_LINK_BRACKET_OUTPUT_CELLS,
    MAX_LINK_BRACKET_WORK,
    CrossingVisit,
    LinkBracketResult,
    LinkComponent,
    LinkComponentOrientationTransport,
    LinkComponentsResult,
    LinkCrossingSignChange,
    LinkDartTransport,
    LinkDiagramMirrorResult,
    LinkingMatrixResult,
    LinkJonesResult,
    LinkOrientationReverseResult,
    LinkState,
    OrientedLinkDiagram,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _admit_components(diagram: OrientedLinkDiagram) -> OrientedLinkDiagram:
    """Canonicalize and enforce the shared envelope for every call path."""

    if not isinstance(diagram, OrientedLinkDiagram):
        _reject(
            "diagram",
            "link_diagram.components.diagram_not_a_link_diagram",
            "component source must be a well-formed oriented link diagram value",
        )
    try:
        admitted = OrientedLinkDiagram.model_validate_json(
            diagram.model_dump_json(warnings=False)
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("diagram",),
            code="link_diagram.components.diagram_shape",
            message="diagram must satisfy the complete oriented-link value contract",
        ) from exc
    if len(admitted.crossings) > 64:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.components.crossings_over_envelope",
            message="link diagram exceeds the 64-crossing envelope",
        )
    return admitted


def link_mirror(diagram: OrientedLinkDiagram) -> LinkDiagramMirrorResult:
    """Mirror a classical link diagram, preserving its dart and arc identities.

    The over- and under-passing strands are exchanged at each crossing and the
    oriented crossing sign is negated. The finite label axes and zero-crossing
    components are unchanged.
    """

    admitted = _admit_components(diagram)
    mirrored = OrientedLinkDiagram(
        crossings=tuple(
            crossing.model_copy(
                update={
                    "over_pair": crossing.under_pair,
                    "under_pair": crossing.over_pair,
                    "sign": -crossing.sign,
                }
            )
            for crossing in admitted.crossings
        ),
        arcs=admitted.arcs,
        free_loops=admitted.free_loops,
    )
    return LinkDiagramMirrorResult(source=admitted, diagram=mirrored)


def link_orientation_reverse(
    diagram: OrientedLinkDiagram,
    component_representatives: tuple[str, ...] = (),
) -> LinkOrientationReverseResult:
    """Reverse selected crossing-bearing components, retaining every dart ID.

    Representatives are source crossing darts.  Crossing-free loops have no
    oriented identity in the current value and therefore cannot be selected.
    """
    admitted = _admit_components(diagram)
    if not isinstance(component_representatives, tuple) or any(
        not isinstance(value, str) or not value or len(value) > 64
        for value in component_representatives
    ):
        _reject(
            "component_representatives",
            "link_diagram.orientation_reverse.representatives_shape",
            "component representatives must be a tuple of valid link labels",
        )
    source_components = link_components(admitted)
    dart_component = {
        dart: component.component_id
        for component in source_components.components
        for dart in component.darts
    }
    crossing_darts = {
        dart for crossing in admitted.crossings for dart in crossing.half_edges
    }
    if len(set(component_representatives)) != len(component_representatives):
        _reject(
            "component_representatives",
            "link_diagram.orientation_reverse.duplicate_representative",
            "component representatives must be distinct",
        )
    selected: set[str] = set()
    for representative in component_representatives:
        component_id = dart_component.get(representative)
        if representative not in crossing_darts or component_id is None:
            _reject(
                "component_representatives",
                "link_diagram.orientation_reverse.untracked_component",
                "each representative must identify a crossing-bearing component; free loops are untracked",
            )
        if component_id in selected:
            _reject(
                "component_representatives",
                "link_diagram.orientation_reverse.duplicate_component",
                "select at most one representative from each source component",
            )
        selected.add(component_id)

    target_arcs = tuple(
        type(arc)(tail=arc.head, head=arc.tail)
        if dart_component[arc.tail] in selected
        else arc
        for arc in admitted.arcs
    )
    target_crossings = tuple(
        crossing.model_copy(
            update={
                "sign": (
                    -crossing.sign
                    if (
                        dart_component[crossing.half_edges[min(crossing.over_pair)]]
                        in selected
                    )
                    != (
                        dart_component[crossing.half_edges[min(crossing.under_pair)]]
                        in selected
                    )
                    else crossing.sign
                )
            }
        )
        for crossing in admitted.crossings
    )
    target = OrientedLinkDiagram(
        crossings=target_crossings, arcs=target_arcs, free_loops=admitted.free_loops
    )
    target_components = link_components(target)
    target_by_darts = {
        frozenset(component.darts): component
        for component in target_components.components
    }
    transport = tuple(
        LinkComponentOrientationTransport(
            source_component_id=component.component_id,
            target_component_id=target_by_darts[
                frozenset(component.darts)
            ].component_id,
            orientation_reversed=component.component_id in selected,
            darts=tuple(
                LinkDartTransport(source_dart=dart, target_dart=dart)
                for dart in sorted(component.darts)
            ),
        )
        for component in source_components.components
    )
    changes = tuple(
        LinkCrossingSignChange(
            crossing_id=source_crossing.crossing_id,
            source_sign=source_crossing.sign,
            target_sign=target_crossing.sign,
        )
        for source_crossing, target_crossing in zip(
            admitted.crossings, target.crossings, strict=True
        )
        if source_crossing.sign != target_crossing.sign
    )
    return LinkOrientationReverseResult(
        source=admitted,
        diagram=target,
        source_components=source_components,
        target_components=target_components,
        component_representatives=component_representatives,
        component_transport=transport,
        crossing_sign_changes=changes,
    )


def _add_term(
    target: dict[int, Fraction], exponent: int, coefficient: Fraction
) -> None:
    if coefficient:
        target[exponent] = target.get(exponent, Fraction(0)) + coefficient
        if target[exponent] == 0:
            del target[exponent]


def _polynomial(terms: dict[int, Fraction]) -> RationalLaurentPolynomial:
    from jacobian.math.polynomials.values import (
        RationalLaurentPolynomial,
        RationalLaurentPolynomialTerm,
    )

    return RationalLaurentPolynomial(
        variables=("A",),
        terms=tuple(
            RationalLaurentPolynomialTerm(
                coefficient=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num=value.numerator, den=value.denominator),
                exponents=(exponent,),
            )
            for exponent, value in sorted(terms.items(), reverse=True)
            if value
        ),
    )


def _union_find(
    size: int,
) -> tuple[list[int], Callable[[int], int], Callable[[int, int], None]]:
    parent = list(range(size))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    return parent, find, union


def _admit_bracket(diagram: OrientedLinkDiagram) -> OrientedLinkDiagram:
    admitted = _admit_components(diagram)
    crossing_count = len(admitted.crossings)
    if crossing_count > MAX_LINK_BRACKET_CROSSINGS:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.bracket.state_bound",
            message=(
                "the exact bracket state family exceeds "
                f"2^{MAX_LINK_BRACKET_CROSSINGS} admitted states"
            ),
        )
    state_count = 1 << crossing_count
    # One work unit is a DSU parent-edge visit or a state-expansion term. With
    # 4c darts, 12c finds per state each visit at most 4c parent edges; the
    # quadratic term dominates these visits and the remaining terms cover
    # initialization and the binomial expansion.
    work_bound = state_count * (
        128 * crossing_count**2 + 8 * crossing_count + 4 * admitted.free_loops + 4
    )
    if work_bound > MAX_LINK_BRACKET_WORK:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.bracket.work_bound",
            message="the conservative bracket state-sum work bound is exceeded",
        )
    # At most two smoothed circles per crossing plus the free loops can occur.
    # The exponent interval then bounds the accumulated Laurent term count.
    delta_power_bound = max(0, 2 * crossing_count + admitted.free_loops - 1)
    polynomial_term_bound = 2 * crossing_count + 4 * delta_power_bound + 1
    # Every state row is bounded by its <=12 binary choices and fixed scalar
    # fields; this counts retained state-row and Laurent-term cells, not bytes.
    output_cells = state_count * 256 + polynomial_term_bound * 256 + 64 * 1024
    if output_cells > MAX_LINK_BRACKET_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.bracket.output_bound",
            message="the conservative bracket result-size bound is exceeded",
        )
    return admitted


def link_bracket(diagram: OrientedLinkDiagram) -> LinkBracketResult:
    """Compute the complete Kauffman bracket state sum in Laurent A."""
    diagram = _admit_bracket(diagram)
    crossings = diagram.crossings
    darts = [dart for crossing in crossings for dart in crossing.half_edges]
    index = {dart: i for i, dart in enumerate(darts)}
    arcs = tuple((index[arc.tail], index[arc.head]) for arc in diagram.arcs)
    terms: dict[int, Fraction] = {}
    states: list[LinkState] = []
    state_choices = product((0, 1), repeat=len(crossings))
    for choices in state_choices:
        _parent, find, union = _union_find(len(darts))
        for left, right in arcs:
            union(left, right)
        for crossing, choice in zip(crossings, choices, strict=True):
            h = [index[item] for item in crossing.half_edges]
            over_uses_even_positions = set(crossing.over_pair) == {0, 2}
            smoothing_index = choice if over_uses_even_positions else 1 - choice
            pairs = ((0, 1, 2, 3), (1, 2, 3, 0))[smoothing_index]
            union(h[pairs[0]], h[pairs[1]])
            union(h[pairs[2]], h[pairs[3]])
        circles = (
            len({find(item) for item in range(len(darts))}) + diagram.free_loops
            if darts
            else diagram.free_loops
        )
        if circles < 1:
            circles = 1
        base_exponent = len(crossings) - 2 * sum(choices)
        # Expand delta^(circles-1), delta = -A^2 - A^-2.
        delta_power = circles - 1
        for negative_choices in range(delta_power + 1):
            exponent = base_exponent + 2 * delta_power - 4 * negative_choices
            coefficient = Fraction(
                (-1) ** delta_power
                * __import__("math").comb(delta_power, negative_choices),
                1,
            )
            _add_term(terms, exponent, coefficient)
        states.append(
            LinkState(
                choices=tuple(choices),
                circle_count=circles,
                exponent=base_exponent,
                coefficient=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num=1, den=1),
            )
        )
    if not states:  # zero-crossing unlink has one empty state
        states.append(
            LinkState(
                choices=(),
                circle_count=max(diagram.free_loops, 1),
                exponent=0,
                coefficient=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num=1, den=1),
            )
        )
        for negative_choices in range(max(diagram.free_loops - 1, 0) + 1):
            power = max(diagram.free_loops - 1, 0)
            _add_term(
                terms,
                2 * power - 4 * negative_choices,
                Fraction(
                    (-1) ** power * __import__("math").comb(power, negative_choices), 1
                ),
            )
    return LinkBracketResult._from_kernel(
        diagram=diagram,
        polynomial=_polynomial(terms),
        states=tuple(states),
        crossing_count=len(crossings),
        state_count=len(states),
    )


def link_jones(diagram: OrientedLinkDiagram) -> LinkJonesResult:
    """Return the writhe-normalized Jones polynomial, represented in A."""
    bracket = link_bracket(diagram)
    diagram = bracket.diagram
    writhe = sum(crossing.sign for crossing in diagram.crossings)
    sign_factor = Fraction(-1 if writhe % 2 else 1, 1)
    terms = {
        term.exponents[0] - 3 * writhe: sign_factor * term.coefficient.as_fraction()
        for term in bracket.polynomial.terms
    }
    return LinkJonesResult._from_kernel(
        diagram=diagram,
        bracket=bracket,
        polynomial=_polynomial(terms),
        writhe=writhe,
    )


def link_linking_matrix(diagram: OrientedLinkDiagram) -> LinkingMatrixResult:
    """Compute the exact oriented linking matrix from signed mixed crossings."""
    components = link_components(diagram)
    diagram = components.diagram
    dart_component = {
        dart: component.component_id
        for component in components.components
        for dart in component.darts
    }
    ids = tuple(component.component_id for component in components.components)
    position = {identifier: index for index, identifier in enumerate(ids)}
    matrix = [[Fraction(0) for _ in ids] for _ in ids]
    for crossing in diagram.crossings:
        over_id = dart_component[crossing.half_edges[min(crossing.over_pair)]]
        under_id = dart_component[crossing.half_edges[min(crossing.under_pair)]]
        if over_id != under_id:
            i, j = position[over_id], position[under_id]
            matrix[i][j] += Fraction(crossing.sign, 2)
            matrix[j][i] += Fraction(crossing.sign, 2)
    return LinkingMatrixResult._from_kernel(
        diagram=diagram,
        component_ids=ids,
        matrix=tuple(
            tuple(
                CanonicalRational(num=value.numerator, den=value.denominator)
                for value in row
            )
            for row in matrix
        ),
    )


def link_components(diagram: OrientedLinkDiagram) -> LinkComponentsResult:
    """Traverse the explicitly oriented diagram cycles in their source direction."""
    diagram = _admit_components(diagram)
    strand_partner: dict[str, str] = {}
    dart_role: dict[str, str] = {}
    crossing_of: dict[str, str] = {}
    for crossing in diagram.crossings:
        over = tuple(crossing.half_edges[i] for i in crossing.over_pair)
        under = tuple(crossing.half_edges[i] for i in crossing.under_pair)
        for left, right, role in (
            (over[0], over[1], "OVER"),
            (under[0], under[1], "UNDER"),
        ):
            strand_partner[left] = right
            strand_partner[right] = left
            dart_role[left] = role
            dart_role[right] = role
        for dart in crossing.half_edges:
            crossing_of[dart] = crossing.crossing_id
    arc_next = {arc.tail: arc.head for arc in diagram.arcs}
    if set(strand_partner) != (set(arc_next) | {arc.head for arc in diagram.arcs}):
        _reject(
            "diagram",
            "link_diagram.components.dart_coverage_failed",
            "every half-edge must have one incoming or outgoing oriented arc",
        )
    visited: set[str] = set()
    components: list[LinkComponent] = []
    for start in sorted(arc_next):
        if start in visited:
            continue
        cycle: list[str] = []
        visits: list[CrossingVisit] = []
        cursor = start
        while cursor not in visited:
            head = arc_next[cursor]
            if cursor in visited or head in visited:
                _reject(
                    "diagram",
                    "link_diagram.components.traversal_not_a_cycle",
                    "oriented arc and strand steps must close into disjoint cycles",
                )
            visited.add(cursor)
            visited.add(head)
            cycle.extend((cursor, head))
            for dart in (cursor, head):
                visits.append(
                    CrossingVisit(
                        crossing_id=crossing_of[dart],
                        role=dart_role[dart],  # type: ignore[arg-type]
                    )
                )
            cursor = strand_partner[head]
        if cursor != start:
            _reject(
                "diagram",
                "link_diagram.components.traversal_not_a_cycle",
                "oriented component traversal must return to its first dart",
            )
        components.append(
            LinkComponent(
                component_id=f"component_{len(components):03d}",
                darts=tuple(cycle),
                visits=tuple(sorted(visits, key=lambda visit: visit.crossing_id)),
                length=len(cycle),
            )
        )
    for loop in range(diagram.free_loops):
        components.append(
            LinkComponent(
                component_id=f"~free_loop_{loop:03d}",
                darts=(f"~free_loop_{loop:03d}:dart",),
                visits=(),
                length=1,
            )
        )
    ordered = tuple(sorted(components, key=lambda component: component.component_id))
    return LinkComponentsResult._from_kernel(diagram=diagram, components=ordered)


__all__ = ["link_bracket", "link_components", "link_jones", "link_linking_matrix"]
