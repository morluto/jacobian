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
    CrossingVisit,
    LinkBracketResult,
    LinkComponent,
    LinkComponentsResult,
    LinkingMatrixResult,
    LinkJonesResult,
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
    if len(admitted.crossings) > 12:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.bracket.state_bound",
            message="the exact bracket state family exceeds 2^12 admitted states",
        )
    return admitted


def link_bracket(diagram: OrientedLinkDiagram) -> LinkBracketResult:
    """Compute the complete Kauffman bracket state sum in Laurent A."""
    diagram = _admit_bracket(diagram)
    crossings = diagram.crossings
    darts = [dart for crossing in crossings for dart in crossing.half_edges]
    index = {dart: i for i, dart in enumerate(darts)}
    arcs = tuple((index[arc.first], index[arc.second]) for arc in diagram.arcs)
    terms: dict[int, Fraction] = {}
    states: list[LinkState] = []
    state_choices = product((0, 1), repeat=len(crossings))
    for choices in state_choices:
        _parent, find, union = _union_find(len(darts))
        for left, right in arcs:
            union(left, right)
        for crossing, choice in zip(crossings, choices, strict=True):
            h = [index[item] for item in crossing.half_edges]
            pairs = ((0, 1, 2, 3), (1, 2, 3, 0))[choice]
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
        over_id = dart_component[crossing.half_edges[crossing.over_pair[0]]]
        under_id = dart_component[crossing.half_edges[crossing.under_pair[0]]]
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
    """Partition a classical oriented link diagram into oriented components.

    At each crossing the over strand joins its over pair and the under
    strand joins its under pair; the arc involution joins half-edges
    between crossings. The two pairings are fixed-point-free involutions,
    so alternating arc/strand steps from the smallest unvisited dart yields
    disjoint cycles covering every half-edge exactly once. Each visit
    records the crossing ID with the strand's OVER/UNDER role. Free loops
    (zero-crossing components) return as empty-dart components.
    """

    diagram = _admit_components(diagram)
    strand_partner: dict[str, str] = {}
    dart_role: dict[str, str] = {}
    for crossing in diagram.crossings:
        darts = crossing.half_edges
        over = tuple(darts[i] for i in crossing.over_pair)
        under = tuple(darts[i] for i in crossing.under_pair)
        strand_partner[over[0]] = over[1]
        strand_partner[over[1]] = over[0]
        strand_partner[under[0]] = under[1]
        strand_partner[under[1]] = under[0]
        dart_role[over[0]] = "OVER"
        dart_role[over[1]] = "OVER"
        dart_role[under[0]] = "UNDER"
        dart_role[under[1]] = "UNDER"
    arc_partner: dict[str, str] = {}
    for arc in diagram.arcs:
        arc_partner[arc.first] = arc.second
        arc_partner[arc.second] = arc.first
    # Replay the dart/half-edge invariant before traversal: every half-edge
    # in exactly one crossing (strand map) and one arc (arc map).
    if set(strand_partner) != set(arc_partner):
        _reject(
            "diagram",
            "link_diagram.components.dart_coverage_failed",
            "every half-edge must lie in exactly one crossing and one arc",
        )
    visited: set[str] = set()
    components: list[LinkComponent] = []
    index = 0
    crossing_of = {
        dart: crossing.crossing_id
        for crossing in diagram.crossings
        for dart in crossing.half_edges
    }
    for start in sorted(strand_partner):
        if start in visited:
            continue
        cycle: list[str] = []
        visits: list[CrossingVisit] = []
        cursor = start
        while cursor not in visited:
            # One arc step lands on the next entry dart, then the strand
            # step crosses to its partner; both darts join the cycle so
            # every half-edge is covered exactly once.
            for dart in (cursor, arc_partner[cursor]):
                visited.add(dart)
                cycle.append(dart)
                visits.append(
                    CrossingVisit(crossing_id=crossing_of[dart], role=dart_role[dart])  # type: ignore[arg-type]
                )
            cursor = strand_partner[arc_partner[cursor]]
        # A cycle closes exactly at its start: the strand/arc involutions
        # are deterministic, so re-entering `visited` elsewhere is a
        # malformed pairing the value contract already excludes.
        if cursor != start:  # pragma: no cover - excluded by value invariant
            _reject(
                "diagram",
                "link_diagram.components.traversal_not_a_cycle",
                "component traversal must close into disjoint cycles",
            )
        components.append(
            LinkComponent(
                component_id=f"component_{index:03d}",
                darts=tuple(cycle),
                visits=tuple(sorted(visits, key=lambda v: v.crossing_id)),
                length=len(cycle),
            )
        )
        index += 1
    for loop in range(diagram.free_loops):
        components.append(
            LinkComponent(
                component_id=f"free_loop_{loop:03d}",
                darts=(f"free_loop_{loop:03d}:dart",),
                visits=(),
                length=1,
            )
        )
    ordered = tuple(sorted(components, key=lambda c: c.component_id))
    # Every diagram arc appears exactly once across all components.
    covered = [dart for component in ordered for dart in component.darts]
    if len(covered) != len(set(covered)):
        _reject(  # pragma: no cover - excluded by traversal construction
            "diagram",
            "link_diagram.components.arc_cover_failed",
            "every diagram arc must appear exactly once",
        )
    return LinkComponentsResult._from_kernel(diagram=diagram, components=ordered)


__all__ = ["link_bracket", "link_components", "link_jones", "link_linking_matrix"]
