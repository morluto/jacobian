"""Native exact link-diagram component traversal."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links._models import (
    CrossingVisit,
    LinkComponent,
    LinkComponentsResult,
    OrientedLinkDiagram,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _admit_components(diagram: OrientedLinkDiagram) -> None:
    """Enforce the shared envelope for native and catalog calls."""

    if not isinstance(diagram, OrientedLinkDiagram):
        _reject(
            "diagram",
            "link_diagram.components.diagram_not_a_link_diagram",
            "component source must be a well-formed oriented link diagram value",
        )
    if len(diagram.crossings) > 64:
        raise OperationResourceAdmissionError(
            location=("diagram",),
            code="link_diagram.components.crossings_over_envelope",
            message="link diagram exceeds the 64-crossing envelope",
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

    _admit_components(diagram)
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
    return LinkComponentsResult._from_kernel(components=ordered)


__all__ = ["link_components"]
