"""Typed contracts for exact classical oriented link diagrams."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalLaurentPolynomial


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"link_diagram.{reason}", message)


MAX_LINK_CROSSINGS = 64
"""Maximum crossings in one admitted link diagram."""

MAX_LINK_LABEL_LENGTH = 64
"""Maximum length of a crossing, half-edge, or component identifier."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


LinkLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_LINK_LABEL_LENGTH, strict=True),
    AfterValidator(_require_scalar_label),
]
"""One crossing, half-edge, or component identifier."""


class LinkCrossing(StrictModel):
    """One classical crossing with cyclic half-edges and over/under data.

    ``half_edges`` are the four distinct local half-edge IDs in the
    published cyclic order. ``over_pair`` and ``under_pair`` are opposite
    pairs in that cyclic order: each is one of ``{0, 2}`` or ``{1, 3}``
    positions, and the two pairs partition the four positions. The strand
    through ``over_pair`` passes over the strand through ``under_pair``.
    """

    crossing_id: LinkLabel
    half_edges: tuple[LinkLabel, LinkLabel, LinkLabel, LinkLabel]
    over_pair: tuple[int, int]
    under_pair: tuple[int, int]
    sign: Literal[-1, 1] = Field(
        default=1, description="Oriented crossing sign used for writhe normalization."
    )

    @model_validator(mode="after")
    def require_opposite_strand_pairs(self) -> Self:
        if len(set(self.half_edges)) != 4:
            raise _validation_error(
                "crossing_half_edges",
                "crossing half-edges must be four distinct IDs",
            )
        pairs = {tuple(sorted(self.over_pair)), tuple(sorted(self.under_pair))}
        if pairs != {(0, 2), (1, 3)}:
            raise _validation_error(
                "crossing_strand_pairs",
                "over/under pairs must be the opposite pairs {0,2} and {1,3}",
            )
        for pair in (self.over_pair, self.under_pair):
            if any(index not in (0, 1, 2, 3) for index in pair) or len(set(pair)) != 2:
                raise _validation_error(
                    "crossing_pair_indices",
                    "strand pairs must be two distinct positions of 0..3",
                )
        return self


class ArcPairing(StrictModel):
    """One diagram arc joining two half-edges between crossings."""

    first: LinkLabel
    second: LinkLabel

    @model_validator(mode="after")
    def require_proper_arc(self) -> Self:
        if self.first == self.second:
            raise _validation_error(
                "arc_loop", "diagram arcs must join two distinct half-edges"
            )
        return self


class OrientedLinkDiagram(StrictModel):
    """A well-formed finite classical oriented link diagram.

    Every half-edge belongs to exactly one crossing and exactly one arc;
    closed zero-crossing components are counted by ``free_loops``. The
    crossing strand pairing together with the arc involution produces
    disjoint oriented component cycles.
    """

    crossings: tuple[LinkCrossing, ...] = Field(
        default=(), max_length=MAX_LINK_CROSSINGS
    )
    arcs: tuple[ArcPairing, ...] = Field(default=())
    free_loops: int = Field(default=0, ge=0, le=MAX_LINK_CROSSINGS)

    @model_validator(mode="after")
    def require_well_formed_diagram(self) -> Self:
        crossing_ids = tuple(crossing.crossing_id for crossing in self.crossings)
        if tuple(sorted(crossing_ids)) != crossing_ids or len(set(crossing_ids)) != len(
            crossing_ids
        ):
            raise _validation_error(
                "crossing_ids",
                "crossing IDs must be unique and strictly ordered",
            )
        darts: list[str] = []
        for crossing in self.crossings:
            darts.extend(crossing.half_edges)
        if len(set(darts)) != len(darts):
            raise _validation_error(
                "dart_coverage",
                "every half-edge must belong to exactly one crossing",
            )
        dart_set = set(darts)
        seen: set[str] = set()
        for arc in self.arcs:
            for end in (arc.first, arc.second):
                if end not in dart_set:
                    raise _validation_error(
                        "arc_coverage",
                        "every arc end must be a crossing half-edge",
                    )
                if end in seen:
                    raise _validation_error(
                        "arc_involution",
                        "every half-edge is paired along exactly one diagram arc",
                    )
                seen.add(end)
        if seen != dart_set:
            raise _validation_error(
                "arc_involution",
                "every half-edge is paired along exactly one diagram arc",
            )
        if not self.crossings and not self.arcs and self.free_loops == 0:
            raise _validation_error(
                "empty_diagram",
                "a link diagram needs a crossing, an arc family, or a free loop",
            )
        return self


CrossingRole = Literal["OVER", "UNDER"]


class CrossingVisit(StrictModel):
    """One component passage through a crossing with its over/under role."""

    crossing_id: LinkLabel
    role: CrossingRole


class LinkComponent(StrictModel):
    """One oriented component as a cyclic dart sequence with crossing roles."""

    component_id: LinkLabel
    darts: tuple[LinkLabel, ...] = Field(min_length=1)
    visits: tuple[CrossingVisit, ...] = Field(default=())
    length: int = Field(ge=1)

    @model_validator(mode="after")
    def require_component_shape(self) -> Self:
        if self.length != len(self.darts):
            raise _validation_error(
                "component_length",
                "component length must equal its dart count",
            )
        if len(set(self.darts)) != len(self.darts):
            raise _validation_error(
                "component_darts",
                "component darts must be distinct within one traversal",
            )
        return self


class LinkState(StrictModel):
    """One complete Kauffman state and its state-circle contribution."""

    choices: tuple[int, ...]
    circle_count: int = Field(ge=1)
    exponent: int
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_binary_choices(self) -> Self:
        if any(
            type(choice) is not int or choice not in (0, 1) for choice in self.choices
        ):
            raise _validation_error(
                "state_choices", "state choices must be binary smoothing indices"
            )
        return self


def _canonical_component_ids(diagram: OrientedLinkDiagram) -> tuple[str, ...]:
    """Return the stable component-axis labels induced by a source diagram."""
    strand_partner: dict[str, str] = {}
    arc_partner: dict[str, str] = {}
    for crossing in diagram.crossings:
        darts = crossing.half_edges
        for pair in (crossing.over_pair, crossing.under_pair):
            left, right = (darts[pair[0]], darts[pair[1]])
            strand_partner[left] = right
            strand_partner[right] = left
    for arc in diagram.arcs:
        arc_partner[arc.first] = arc.second
        arc_partner[arc.second] = arc.first
    visited: set[str] = set()
    component_count = 0
    for start in sorted(strand_partner):
        if start in visited:
            continue
        cursor = start
        while cursor not in visited:
            visited.add(cursor)
            next_dart = arc_partner[cursor]
            visited.add(next_dart)
            cursor = strand_partner[next_dart]
        component_count += 1
    labels = [f"component_{index:03d}" for index in range(component_count)]
    labels.extend(f"free_loop_{index:03d}" for index in range(diagram.free_loops))
    return tuple(sorted(labels))


class LinkBracketRequest(StrictModel):
    diagram: OrientedLinkDiagram = Field(
        description=(
            "A well-formed oriented diagram with at most 12 crossings for the "
            "complete 2^crossings state-sum envelope (the value carrier permits 64)."
        )
    )


class LinkBracketResult(StrictModel):
    """Exact bounded Kauffman bracket with every state retained."""

    diagram: OrientedLinkDiagram
    polynomial: RationalLaurentPolynomial
    states: tuple[LinkState, ...]
    crossing_count: int = Field(ge=0)
    state_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_state_axis(self) -> Self:
        from itertools import product

        crossing_count = len(self.diagram.crossings)
        expected_states = tuple(product((0, 1), repeat=crossing_count))
        if (
            self.crossing_count != crossing_count
            or self.state_count != len(self.states)
            or self.state_count != len(expected_states)
        ):
            raise _validation_error(
                "bracket_state_axis",
                "state metadata must match the complete source crossing axis",
            )
        choices = tuple(state.choices for state in self.states)
        if choices != expected_states or len(set(choices)) != len(choices):
            raise _validation_error(
                "bracket_state_exhaustiveness",
                "bracket states must be the unique exhaustive binary state family",
            )
        from jacobian.math.topology.links.operations import link_bracket

        if link_bracket(self.diagram) != self:
            raise _validation_error(
                "bracket_source_relation",
                "state circles, exponents, coefficients, and polynomial must equal the source state sum",
            )
        return self

    @property
    def bracket(self) -> RationalLaurentPolynomial:
        return self.polynomial

    @property
    def state_circles(self) -> tuple[int, ...]:
        return tuple(state.circle_count for state in self.states)

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class LinkJonesRequest(StrictModel):
    diagram: OrientedLinkDiagram = Field(
        description=(
            "A well-formed oriented diagram; exact bracket work is limited to "
            "12 crossings even though the diagram carrier permits 64."
        )
    )


class LinkJonesResult(StrictModel):
    """Jones normalization of the exact bracket in the Laurent variable A."""

    diagram: OrientedLinkDiagram
    bracket: LinkBracketResult
    polynomial: RationalLaurentPolynomial
    writhe: int
    normalization: Literal["(-A)^-3w_BRACKET"] = "(-A)^-3w_BRACKET"

    @model_validator(mode="after")
    def require_bracket_source(self) -> Self:
        if self.bracket.diagram != self.diagram:
            raise _validation_error(
                "jones_bracket_source", "nested bracket must bind the outer diagram"
            )
        if self.writhe != sum(crossing.sign for crossing in self.diagram.crossings):
            raise _validation_error(
                "jones_writhe_source", "writhe must equal the retained crossing signs"
            )
        from jacobian.math.topology.links.operations import link_jones

        if link_jones(self.diagram) != self:
            raise _validation_error(
                "jones_source_relation",
                "Jones polynomial must equal the writhe normalization of the source bracket",
            )
        return self

    @property
    def jones(self) -> RationalLaurentPolynomial:
        return self.polynomial

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class LinkingMatrixResult(StrictModel):
    """Oriented pairwise linking numbers, source-bound to a link diagram."""

    diagram: OrientedLinkDiagram
    component_ids: tuple[str, ...]
    matrix: tuple[tuple[CanonicalRational, ...], ...]

    @model_validator(mode="after")
    def require_source_component_axis(self) -> Self:
        expected = _canonical_component_ids(self.diagram)
        if self.component_ids != expected:
            raise _validation_error(
                "linking_component_axis",
                "component IDs must be the canonical source component axis",
            )
        size = len(expected)
        if len(self.matrix) != size or any(len(row) != size for row in self.matrix):
            raise _validation_error(
                "linking_matrix_shape",
                "linking matrix must be square on the source axis",
            )
        if any(self.matrix[index][index].as_fraction() != 0 for index in range(size)):
            raise _validation_error(
                "linking_matrix_diagonal",
                "linking matrix diagonal entries must vanish",
            )
        if any(
            self.matrix[left][right] != self.matrix[right][left]
            for left in range(size)
            for right in range(size)
        ):
            raise _validation_error(
                "linking_matrix_symmetry",
                "linking matrix must be symmetric on the source component axis",
            )
        from jacobian.math.topology.links.operations import link_linking_matrix

        if link_linking_matrix(self.diagram) != self:
            raise _validation_error(
                "linking_matrix_source_relation",
                "linking entries must equal signed mixed-crossing sums",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class LinkComponentsRequest(StrictModel):
    """Partition a well-formed classical oriented link diagram into components."""

    diagram: OrientedLinkDiagram = Field(
        description=(
            "Well-formed classical oriented link diagram: every half-edge in "
            "exactly one crossing and one arc; traversal gives disjoint "
            "cycles covering every diagram arc."
        )
    )


class LinkComponentsResult(StrictModel):
    """Complete source-bound oriented component partition with crossing roles."""

    diagram: OrientedLinkDiagram
    components: tuple[LinkComponent, ...] = Field(min_length=1)
    component_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_partition_shape(self) -> Self:
        if self.component_count != len(self.components):
            raise _validation_error(
                "component_count",
                "component count must equal the component family length",
            )
        ids = tuple(component.component_id for component in self.components)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise _validation_error(
                "component_ids",
                "component IDs must be unique and strictly ordered",
            )
        expected_ids = _canonical_component_ids(self.diagram)
        if ids != expected_ids:
            raise _validation_error(
                "component_source_axis",
                "component IDs must equal the retained diagram component axis",
            )
        crossing_by_dart = {
            dart: crossing
            for crossing in self.diagram.crossings
            for dart in crossing.half_edges
        }
        expected_darts = set(crossing_by_dart)
        expected_darts.update(
            f"free_loop_{index:03d}:dart" for index in range(self.diagram.free_loops)
        )
        covered: list[str] = []
        for component in self.components:
            if any(dart not in expected_darts for dart in component.darts):
                raise _validation_error(
                    "component_source_darts",
                    "component darts must belong to the retained diagram",
                )
            covered.extend(component.darts)
            expected_visits = sorted(
                (
                    crossing_by_dart[dart].crossing_id,
                    "OVER"
                    if dart
                    in {
                        crossing_by_dart[dart].half_edges[index]
                        for index in crossing_by_dart[dart].over_pair
                    }
                    else "UNDER",
                )
                for dart in component.darts
                if dart in crossing_by_dart
            )
            actual_visits = sorted(
                (visit.crossing_id, visit.role) for visit in component.visits
            )
            if actual_visits != expected_visits:
                raise _validation_error(
                    "component_source_visits",
                    "component crossing visits must bind source darts and roles",
                )
        if sorted(covered) != sorted(expected_darts) or len(covered) != len(
            set(covered)
        ):
            raise _validation_error(
                "component_source_partition",
                "components must partition every retained diagram dart exactly once",
            )
        from jacobian.math.topology.links.operations import link_components

        if link_components(self.diagram) != self:
            raise _validation_error(
                "component_source_cycles",
                "component darts must equal the canonical alternating arc/strand cycles",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, diagram: OrientedLinkDiagram, components: tuple[LinkComponent, ...]
    ) -> Self:
        """Build a trusted kernel outcome without replaying its traversal."""

        return cls.model_construct(
            diagram=diagram, components=components, component_count=len(components)
        )


__all__ = [
    "MAX_LINK_CROSSINGS",
    "MAX_LINK_LABEL_LENGTH",
    "ArcPairing",
    "CrossingRole",
    "CrossingVisit",
    "LinkBracketRequest",
    "LinkBracketResult",
    "LinkComponent",
    "LinkComponentsRequest",
    "LinkComponentsResult",
    "LinkCrossing",
    "LinkJonesRequest",
    "LinkJonesResult",
    "LinkLabel",
    "LinkState",
    "LinkingMatrixResult",
    "OrientedLinkDiagram",
]
