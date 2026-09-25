"""Typed contracts for exact classical oriented link diagrams."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalLaurentPolynomial
from jacobian.math.topology.links._diagram_validation import (
    validate_crossing_orientations,
    validate_dart_axes,
    validate_sphere_embedding,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"link_diagram.{reason}", message)


MAX_LINK_CROSSINGS = 64
"""Maximum crossings in one admitted link diagram."""

MAX_LINK_LABEL_LENGTH = 64
"""Maximum length of a crossing, half-edge, or component identifier."""

MAX_LINK_BRACKET_CROSSINGS = 12
"""Largest complete bracket/Jones state family: 2^12 states."""

MAX_LINK_BRACKET_WORK = 80_000_000
"""Conservative work-unit ceiling for one bracket state sum."""

MAX_LINK_BRACKET_OUTPUT_CELLS = 4 * 1024 * 1024
"""Conservative materialization-cell ceiling for one bracket/Jones result.

The bracket output bound counts retained state rows and Laurent term cells
(``state_count * 256 + polynomial_term_bound * 256`` in ``operations.py``), not
transport bytes; this ceiling dominates the worst admissible 12-crossing state
sum, so no valid request is rejected by it.
"""


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

    ``half_edges`` are the four distinct local half-edge IDs in counterclockwise
    cyclic order under the canonical oriented-planar diagram contract. ``over_pair`` and ``under_pair`` are opposite
    pairs in that cyclic order: each is one of ``{0, 2}`` or ``{1, 3}``
    positions, and the two pairs partition the four positions. The strand
    through ``over_pair`` passes over the strand through ``under_pair``.
    """

    crossing_id: LinkLabel
    half_edges: tuple[LinkLabel, LinkLabel, LinkLabel, LinkLabel]
    over_pair: tuple[int, int]
    under_pair: tuple[int, int]
    sign: Literal[-1, 1] = Field(
        description="Crossing sign checked against directed component orientation."
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


class OrientedDiagramArc(StrictModel):
    """An oriented diagram arc from its departing crossing to its arriving one."""

    tail: LinkLabel
    head: LinkLabel

    @model_validator(mode="after")
    def require_proper_arc(self) -> Self:
        if self.tail == self.head:
            raise _validation_error(
                "arc_loop", "diagram arcs must join two distinct half-edges"
            )
        return self


class OrientedLinkDiagram(StrictModel):
    """Bounded classical diagram with orientation and a sphere rotation system.

    Each crossing's half-edge order is counterclockwise. Arcs point from the
    departing (tail) dart to the arriving (head) dart. This direction orients
    every component. Crossing signs are required and checked from the directed
    tangents and over/under pairs. The cyclic rotation and arc involution must
    satisfy the sphere Euler identity on each projection component.
    """

    crossings: tuple[LinkCrossing, ...] = Field(
        default=(), max_length=MAX_LINK_CROSSINGS
    )
    arcs: tuple[OrientedDiagramArc, ...] = Field(
        default=(), max_length=2 * MAX_LINK_CROSSINGS
    )
    free_loops: int = Field(default=0, ge=0, le=MAX_LINK_CROSSINGS)

    @model_validator(mode="after")
    def require_oriented_planar_diagram(self) -> Self:
        if not self.crossings:
            if self.arcs:
                raise _validation_error(
                    "crossing_free_arcs", "crossing-free components use free_loops"
                )
            if not self.free_loops:
                raise _validation_error(
                    "empty_diagram", "a link diagram needs a crossing or a free loop"
                )
            return self
        try:
            dart_owner, partner, tails, heads = validate_dart_axes(
                self.crossings, self.arcs
            )
            validate_crossing_orientations(self.crossings, tails, heads)
            validate_sphere_embedding(self.crossings, self.arcs, dart_owner, partner)
        except ValueError as exc:
            raise _validation_error("oriented_planar_shape", str(exc)) from exc
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


class LinkBracketRequest(StrictModel):
    diagram: OrientedLinkDiagram = Field(
        description=(
            "A well-formed oriented diagram with at most 12 crossings, a "
            "conservative 80,000,000-work-unit state-sum bound, and a 4 MiB "
            "estimated result ceiling (the value carrier permits 64 crossings)."
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
        crossing_count = len(self.diagram.crossings)
        expected_state_count = 1 << crossing_count
        if (
            self.crossing_count != crossing_count
            or self.state_count != len(self.states)
            or self.state_count != expected_state_count
        ):
            raise _validation_error(
                "bracket_state_axis",
                "state metadata must match the complete source crossing axis",
            )
        choices = tuple(state.choices for state in self.states)
        if (
            any(len(choice) != crossing_count for choice in choices)
            or len(set(choices)) != expected_state_count
        ):
            raise _validation_error(
                "bracket_state_exhaustiveness",
                "bracket states must be the unique exhaustive binary state family",
            )
        if any(
            state.exponent != crossing_count - 2 * sum(state.choices)
            or state.coefficient.as_fraction() != 1
            for state in self.states
        ):
            raise _validation_error(
                "bracket_state_metadata",
                "state exponents and coefficients must bind their smoothing choices",
            )
        if self.polynomial.variables != ("A",):
            raise _validation_error(
                "bracket_polynomial_axis",
                "Kauffman bracket terms must use the retained Laurent variable A",
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
            "A well-formed oriented diagram; the writhe-normalized Jones "
            "state sum admits at most 12 crossings (2^12 states), a conservative "
            "80,000,000-work-unit bound, and a 4 MiB estimated result ceiling, "
            "even though the diagram carrier permits 64 crossings."
        )
    )


class LinkJonesResult(StrictModel):
    """Jones polynomial in A, normalized by ``(-A)^(-3w) <D>(A)``.

    The ordinary variable is ``t = A^(-4)``. This A-variable form keeps
    integral Laurent exponents for links without introducing square roots.
    """

    diagram: OrientedLinkDiagram
    bracket: LinkBracketResult
    polynomial: RationalLaurentPolynomial
    writhe: int
    normalization: Literal["(-A)^(-3w(D))*<D>(A); t=A^(-4); output in A"] = (
        "(-A)^(-3w(D))*<D>(A); t=A^(-4); output in A"
    )

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
        if self.polynomial.variables != ("A",):
            raise _validation_error(
                "jones_polynomial_axis",
                "Jones terms must use the retained Laurent variable A",
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
        if not self.component_ids or self.component_ids != tuple(
            sorted(set(self.component_ids))
        ):
            raise _validation_error(
                "linking_component_axis",
                "component IDs must be unique and strictly ordered",
            )
        size = len(self.component_ids)
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


class LinkDiagramMirrorRequest(StrictModel):
    """Mirror every crossing of one oriented classical link diagram."""

    diagram: OrientedLinkDiagram


class LinkDiagramMirrorResult(StrictModel):
    """A mirrored diagram retaining the identity transport of its labels."""

    source: OrientedLinkDiagram
    diagram: OrientedLinkDiagram

    @model_validator(mode="after")
    def require_mirror_transport(self) -> Self:
        if self.source.free_loops != self.diagram.free_loops:
            raise _validation_error(
                "mirror_transport", "mirroring must preserve zero-crossing components"
            )
        if self.source.arcs != self.diagram.arcs:
            raise _validation_error(
                "mirror_transport", "mirroring must preserve the arc pairing"
            )
        if len(self.source.crossings) != len(self.diagram.crossings):
            raise _validation_error(
                "mirror_transport", "mirroring must preserve the crossing axis"
            )
        for source, mirrored in zip(
            self.source.crossings, self.diagram.crossings, strict=True
        ):
            if (
                source.crossing_id != mirrored.crossing_id
                or source.half_edges != mirrored.half_edges
                or source.over_pair != mirrored.under_pair
                or source.under_pair != mirrored.over_pair
                or source.sign != -mirrored.sign
            ):
                raise _validation_error(
                    "mirror_transport",
                    "mirroring must swap strand roles and negate crossing signs",
                )
        return self


class LinkOrientationReverseRequest(StrictModel):
    """Reverse explicitly selected crossing-bearing components by dart labels."""

    diagram: OrientedLinkDiagram
    component_representatives: tuple[LinkLabel, ...] = Field(
        default=(),
        max_length=2 * MAX_LINK_CROSSINGS,
        description=(
            "One crossing half-edge from each component to reverse. Representatives "
            "must lie on distinct source components; crossing-free loops have no "
            "representative in the current diagram value."
        ),
    )

    @model_validator(mode="after")
    def require_distinct_representatives(self) -> Self:
        if len(set(self.component_representatives)) != len(
            self.component_representatives
        ):
            raise _validation_error(
                "orientation_reverse_representatives",
                "component representatives must be distinct",
            )
        return self


class LinkDartTransport(StrictModel):
    """Identity transport of a dart through component orientation reversal."""

    source_dart: LinkLabel
    target_dart: LinkLabel


class LinkComponentOrientationTransport(StrictModel):
    """Source-to-target component identity and complete dart transport."""

    source_component_id: LinkLabel
    target_component_id: LinkLabel
    orientation_reversed: bool
    darts: tuple[LinkDartTransport, ...]


class LinkCrossingSignChange(StrictModel):
    crossing_id: LinkLabel
    source_sign: Literal[-1, 1]
    target_sign: Literal[-1, 1]


class LinkOrientationReverseResult(StrictModel):
    """Source-bound selected orientation reversal with complete component map."""

    source: OrientedLinkDiagram
    diagram: OrientedLinkDiagram
    source_components: LinkComponentsResult
    target_components: LinkComponentsResult
    component_representatives: tuple[LinkLabel, ...]
    component_transport: tuple[LinkComponentOrientationTransport, ...]
    crossing_sign_changes: tuple[LinkCrossingSignChange, ...]

    @model_validator(mode="after")
    def require_orientation_transport(self) -> Self:  # noqa: C901
        if self.source_components.diagram != self.source:
            raise _validation_error(
                "orientation_reverse_source_components",
                "source component partition must bind the source diagram",
            )
        if self.target_components.diagram != self.diagram:
            raise _validation_error(
                "orientation_reverse_target_components",
                "target component partition must bind the transformed diagram",
            )
        if self.source.free_loops != self.diagram.free_loops:
            raise _validation_error(
                "orientation_reverse_free_loops",
                "orientation reversal must retain free-loop count",
            )
        source_by_id = {
            component.component_id: component
            for component in self.source_components.components
        }
        target_by_id = {
            component.component_id: component
            for component in self.target_components.components
        }
        if tuple(row.source_component_id for row in self.component_transport) != tuple(
            sorted(source_by_id)
        ):
            raise _validation_error(
                "orientation_reverse_component_axis",
                "transport must cover the complete ordered source component axis",
            )
        selected_ids: set[str] = set()
        representative_components: list[str] = []
        source_dart_component: dict[str, str] = {}
        for component in self.source_components.components:
            for dart in component.darts:
                source_dart_component[dart] = component.component_id
        for representative in self.component_representatives:
            component_id = source_dart_component.get(representative)
            if component_id is None or representative not in {
                dart
                for crossing in self.source.crossings
                for dart in crossing.half_edges
            }:
                raise _validation_error(
                    "orientation_reverse_component_representative",
                    "each representative must be a crossing dart in the source diagram",
                )
            representative_components.append(component_id)
        if len(set(representative_components)) != len(representative_components):
            raise _validation_error(
                "orientation_reverse_component_selection",
                "select at most one representative from each source component",
            )
        selected_ids.update(representative_components)

        target_arc_directions = {
            frozenset((arc.tail, arc.head)): (arc.tail, arc.head)
            for arc in self.diagram.arcs
        }
        if len(target_arc_directions) != len(self.diagram.arcs):
            raise _validation_error(
                "orientation_reverse_arc_axis", "target arcs must have unique endpoints"
            )
        for row in self.component_transport:
            source_component = source_by_id[row.source_component_id]
            target_component = target_by_id.get(row.target_component_id)
            if target_component is None:
                raise _validation_error(
                    "orientation_reverse_target_component",
                    "every source component must map to one target component",
                )
            source_darts = tuple(sorted(source_component.darts))
            if tuple(item.source_dart for item in row.darts) != source_darts:
                raise _validation_error(
                    "orientation_reverse_dart_axis",
                    "component transport must cover every source dart exactly once",
                )
            if any(item.target_dart != item.source_dart for item in row.darts):
                raise _validation_error(
                    "orientation_reverse_dart_identity",
                    "orientation reversal must retain each dart identity",
                )
            if set(source_darts) != set(target_component.darts):
                raise _validation_error(
                    "orientation_reverse_component_map",
                    "component transport must preserve the complete dart subset",
                )
            if row.orientation_reversed != (row.source_component_id in selected_ids):
                raise _validation_error(
                    "orientation_reverse_selection_map",
                    "transport reversal flags must match selected source components",
                )
        source_arc_component = {
            arc.tail: source_dart_component[arc.tail] for arc in self.source.arcs
        }
        for arc in self.source.arcs:
            component_id = source_arc_component[arc.tail]
            expected = (
                (arc.head, arc.tail)
                if component_id in selected_ids
                else (arc.tail, arc.head)
            )
            if target_arc_directions.get(frozenset((arc.tail, arc.head))) != expected:
                raise _validation_error(
                    "orientation_reverse_arc_transport",
                    "only arcs of selected components may reverse direction",
                )

        expected_changes: list[LinkCrossingSignChange] = []
        for source_crossing, target_crossing in zip(
            self.source.crossings, self.diagram.crossings, strict=True
        ):
            if (
                source_crossing.crossing_id != target_crossing.crossing_id
                or source_crossing.half_edges != target_crossing.half_edges
                or source_crossing.over_pair != target_crossing.over_pair
                or source_crossing.under_pair != target_crossing.under_pair
            ):
                raise _validation_error(
                    "orientation_reverse_crossing_transport",
                    "orientation reversal must preserve crossing and strand identities",
                )
            over_component = source_dart_component[
                source_crossing.half_edges[source_crossing.over_pair[0]]
            ]
            under_component = source_dart_component[
                source_crossing.half_edges[source_crossing.under_pair[0]]
            ]
            sign_changes = (over_component in selected_ids) != (
                under_component in selected_ids
            )
            expected_sign = (
                -source_crossing.sign if sign_changes else source_crossing.sign
            )
            if target_crossing.sign != expected_sign:
                raise _validation_error(
                    "orientation_reverse_crossing_sign",
                    "crossing signs change exactly when one strand is reversed",
                )
            if sign_changes:
                expected_changes.append(
                    LinkCrossingSignChange(
                        crossing_id=source_crossing.crossing_id,
                        source_sign=source_crossing.sign,
                        target_sign=target_crossing.sign,
                    )
                )
        if self.crossing_sign_changes != tuple(expected_changes):
            raise _validation_error(
                "orientation_reverse_sign_axis",
                "crossing sign transport must list every and only changed sign",
            )
        return self


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
    "CrossingRole",
    "CrossingVisit",
    "LinkBracketRequest",
    "LinkBracketResult",
    "LinkComponent",
    "LinkComponentOrientationTransport",
    "LinkComponentsRequest",
    "LinkComponentsResult",
    "LinkCrossing",
    "LinkCrossingSignChange",
    "LinkDartTransport",
    "LinkJonesRequest",
    "LinkJonesResult",
    "LinkLabel",
    "LinkOrientationReverseRequest",
    "LinkOrientationReverseResult",
    "LinkState",
    "LinkingMatrixResult",
    "OrientedDiagramArc",
    "OrientedLinkDiagram",
]
