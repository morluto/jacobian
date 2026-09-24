"""Typed braid and Wirtinger contracts for classical link diagrams."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.polynomials.values import RationalLaurentPolynomial
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupPresentation,
    FiniteGroupWord,
)
from jacobian.math.topology.links._models import (
    MAX_LINK_CROSSINGS,
    LinkComponentsResult,
    LinkLabel,
    OrientedLinkDiagram,
)

MAX_BRAID_STRANDS = 32
MAX_BRAID_WORD_LENGTH = MAX_LINK_CROSSINGS
MAX_WIRTINGER_GENERATORS = 64
MAX_STATE_CIRCLE_CROSSINGS = MAX_LINK_CROSSINGS
MAX_STATE_CIRCLE_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_CONWAY_CENTERED_DEGREE = 64
MAX_CONWAY_COEFFICIENT_DIGITS = 4_096
MAX_CONWAY_OUTPUT_BYTES = 1024 * 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"link_diagram.{reason}", message)


class AlexanderPolynomialRequest(StrictModel):
    diagram: OrientedLinkDiagram


class AlexanderPolynomialResult(StrictModel):
    """A knot's primitive one-variable Alexander polynomial."""

    diagram: OrientedLinkDiagram
    polynomial: RationalLaurentPolynomial
    normalization: Literal["primitive_shifted_nonnegative_positive_constant"] = (
        "primitive_shifted_nonnegative_positive_constant"
    )

    @model_validator(mode="after")
    def require_polynomial_context(self) -> Self:
        if self.polynomial.variables != ("t",):
            raise _validation_error(
                "alexander_polynomial_variable",
                "Alexander polynomial must use the canonical one-variable axis t",
            )
        return self


class ConwayPolynomialRequest(StrictModel):
    diagram: OrientedLinkDiagram


class ConwayPolynomialResult(StrictModel):
    """Knot Conway polynomial bound to its exact Alexander and diagram source."""

    alexander: AlexanderPolynomialResult
    polynomial: RationalLaurentPolynomial
    normalization: Literal["Delta(t)=nabla(t^(1/2)-t^(-1/2)); Delta(1)=1"] = (
        "Delta(t)=nabla(t^(1/2)-t^(-1/2)); Delta(1)=1"
    )

    @model_validator(mode="after")
    def require_conway_polynomial_context(self) -> Self:
        if self.polynomial.variables != ("z",):
            raise _validation_error(
                "conway_polynomial_variable",
                "Conway polynomial must use the canonical variable z",
            )
        coefficients: dict[int, int] = {}
        for term in self.polynomial.terms:
            coefficient = term.coefficient.as_fraction()
            exponent = term.exponents[0]
            if coefficient.denominator != 1 or exponent < 0 or exponent % 2:
                raise _validation_error(
                    "conway_polynomial_support",
                    "knot Conway terms must have integral coefficients and nonnegative even exponents",
                )
            coefficients[exponent] = coefficient.numerator
        if coefficients.get(0, 0) != 1:
            raise _validation_error(
                "conway_polynomial_augmentation",
                "the normalized knot Conway polynomial must have constant term one",
            )
        return self


class LinkCrossingProfileRequest(StrictModel):
    diagram: OrientedLinkDiagram


class LinkCrossingProfileEntry(StrictModel):
    """Sign and ordered strand-component roles at one source crossing."""

    crossing_id: LinkLabel
    sign: Literal[-1, 1]
    over_component_id: LinkLabel
    under_component_id: LinkLabel


class LinkCrossingProfileResult(StrictModel):
    """Complete source-axis crossing roles, component pair, and writhe."""

    components: LinkComponentsResult
    crossings: tuple[LinkCrossingProfileEntry, ...] = Field(
        max_length=MAX_LINK_CROSSINGS
    )
    writhe: StrictInt

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        diagram = self.components.diagram
        if tuple(row.crossing_id for row in self.crossings) != tuple(
            crossing.crossing_id for crossing in diagram.crossings
        ):
            raise _validation_error(
                "crossing_profile_axis",
                "profile rows must retain the complete source crossing order",
            )
        component_ids = {
            component.component_id for component in self.components.components
        }
        by_crossing_role: dict[tuple[str, str], set[str]] = {}
        for component in self.components.components:
            for visit in component.visits:
                by_crossing_role.setdefault((visit.crossing_id, visit.role), set()).add(
                    component.component_id
                )
        for row, crossing in zip(self.crossings, diagram.crossings, strict=True):
            if row.sign != crossing.sign:
                raise _validation_error(
                    "crossing_profile_sign",
                    "profile sign must equal source crossing sign",
                )
            if (
                row.over_component_id not in component_ids
                or row.under_component_id not in component_ids
                or by_crossing_role.get((row.crossing_id, "OVER"))
                != {row.over_component_id}
                or by_crossing_role.get((row.crossing_id, "UNDER"))
                != {row.under_component_id}
            ):
                raise _validation_error(
                    "crossing_profile_roles",
                    "over/under component IDs must match every retained crossing visit",
                )
        if self.writhe != sum(row.sign for row in self.crossings):
            raise _validation_error(
                "crossing_profile_writhe",
                "writhe must equal the complete signed profile",
            )
        return self


SmoothingChoice = Literal["A", "B"]


class LinkDiagramSmoothingState(StrictModel):
    """A complete A/B smoothing choice on the source crossing axis.

    In the canonical counterclockwise dart convention, A pairs adjacent
    positions (0,1) and (2,3) when the over strand occupies (0,2), and uses
    the other adjacent pairing when the over strand occupies (1,3). B uses
    the complementary pairing.
    """

    diagram: OrientedLinkDiagram
    choices: tuple[SmoothingChoice, ...]

    @model_validator(mode="after")
    def require_complete_crossing_axis(self) -> Self:
        if len(self.choices) != len(self.diagram.crossings):
            raise _validation_error(
                "smoothing_state_axis",
                "smoothing choices must cover the complete source crossing axis",
            )
        return self


class LinkStateCirclesRequest(StrictModel):
    state: LinkDiagramSmoothingState


class LinkSmoothedCircle(StrictModel):
    """One canonically oriented cyclic sequence of source darts."""

    darts: tuple[LinkLabel, ...] = Field(default=())


class LinkStateCirclesResult(StrictModel):
    """The exact circle partition induced by one complete smoothing state."""

    state: LinkDiagramSmoothingState
    circles: tuple[LinkSmoothedCircle, ...]
    circle_count: StrictInt = Field(ge=1, le=2 * MAX_LINK_CROSSINGS)

    @model_validator(mode="after")
    def require_complete_cyclic_partition(self) -> Self:
        diagram = self.state.diagram
        expected = tuple(
            dart for crossing in diagram.crossings for dart in crossing.half_edges
        )
        flattened = tuple(dart for circle in self.circles for dart in circle.darts)
        if (
            flattened
            and (
                len(flattened) != len(set(flattened)) or set(flattened) != set(expected)
            )
        ) or (not flattened and diagram.crossings):
            raise _validation_error(
                "state_circle_partition",
                "smoothed circles must partition every source dart exactly once",
            )
        if diagram.crossings and any(not circle.darts for circle in self.circles):
            raise _validation_error(
                "state_circle_empty",
                "a crossing-bearing state circle must contain darts",
            )
        if not diagram.crossings and any(circle.darts for circle in self.circles):
            raise _validation_error(
                "state_circle_free_loop", "crossing-free circles have no dart labels"
            )
        if any(
            circle.darts and circle.darts[0] != min(circle.darts)
            for circle in self.circles
        ):
            raise _validation_error(
                "state_circle_rotation",
                "each cyclic dart sequence must start at its least dart",
            )
        expected_count = len(self.circles) if diagram.crossings else diagram.free_loops
        if self.circle_count != expected_count or len(self.circles) != expected_count:
            raise _validation_error(
                "state_circle_count", "circle count must equal the retained circle axis"
            )
        if self.circles != tuple(sorted(self.circles, key=lambda circle: circle.darts)):
            raise _validation_error(
                "state_circle_order", "circles must use canonical lexicographic order"
            )
        return self


class CheckerboardRegion(StrictModel):
    region_id: LinkLabel
    boundary_darts: tuple[LinkLabel, ...] = Field(
        min_length=1, max_length=4 * MAX_LINK_CROSSINGS
    )
    shaded: bool


class LinkBlackboardEdge(StrictModel):
    """One crossing edge in the canonical shaded-region Tait graph.

    Equal endpoints are permitted: they are loops. Repeated endpoint pairs
    retain distinct crossing IDs and therefore remain parallel edges.
    """

    crossing_id: LinkLabel
    first_region_id: LinkLabel
    second_region_id: LinkLabel
    first_corner_index: StrictInt = Field(ge=0, le=3)
    second_corner_index: StrictInt = Field(ge=0, le=3)
    tait_sign: Literal[-1, 1]

    @model_validator(mode="after")
    def require_opposite_corner_transport(self) -> Self:
        if (
            self.first_corner_index >= self.second_corner_index
            or (self.first_corner_index - self.second_corner_index) % 2
        ):
            raise _validation_error(
                "blackboard_edge_corners",
                "edge endpoints must retain the ordered opposite shaded corners",
            )
        return self


class LinkBlackboardGraph(StrictModel):
    """Source-bound signed Tait graph and its complete planar region data."""

    diagram: OrientedLinkDiagram
    regions: tuple[CheckerboardRegion, ...] = Field(
        min_length=2, max_length=MAX_LINK_CROSSINGS + 2
    )
    shaded_region_ids: tuple[LinkLabel, ...] = Field(
        min_length=1, max_length=MAX_LINK_CROSSINGS + 1
    )
    edges: tuple[LinkBlackboardEdge, ...] = Field(max_length=MAX_LINK_CROSSINGS)

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if (
            not self.diagram.crossings
            or self.diagram.free_loops
            or len(self.regions) != len(self.diagram.crossings) + 2
        ):
            raise _validation_error(
                "blackboard_diagram_axis",
                "graph must retain a nonempty connected crossing projection and its sphere face count",
            )
        region_ids = tuple(region.region_id for region in self.regions)
        if region_ids != tuple(
            f"region_{index:03d}" for index in range(len(region_ids))
        ):
            raise _validation_error(
                "blackboard_region_axis", "region IDs must be in canonical face order"
            )
        if len(set(region_ids)) != len(region_ids):
            raise _validation_error(
                "blackboard_region_ids", "region IDs must be unique"
            )
        boundaries = tuple(region.boundary_darts for region in self.regions)
        darts = tuple(
            dart for crossing in self.diagram.crossings for dart in crossing.half_edges
        )
        covered = tuple(
            dart for region in self.regions for dart in region.boundary_darts
        )
        if sorted(covered) != sorted(darts) or len(covered) != len(set(covered)):
            raise _validation_error(
                "blackboard_region_darts",
                "region boundaries must partition source darts",
            )
        rotation_successor = {
            dart: crossing.half_edges[(index + 1) % 4]
            for crossing in self.diagram.crossings
            for index, dart in enumerate(crossing.half_edges)
        }
        arc_partner = {
            dart: partner
            for arc in self.diagram.arcs
            for dart, partner in ((arc.tail, arc.head), (arc.head, arc.tail))
        }
        face_successor = {dart: rotation_successor[arc_partner[dart]] for dart in darts}
        if any(
            boundary[0] != min(boundary)
            or any(
                face_successor[left] != right
                for left, right in zip(
                    boundary, boundary[1:] + boundary[:1], strict=True
                )
            )
            for boundary in boundaries
        ):
            raise _validation_error(
                "blackboard_region_face_cycles",
                "each boundary must be the canonical face cycle of the source diagram",
            )
        if boundaries != tuple(sorted(boundaries)):
            raise _validation_error(
                "blackboard_region_order", "regions must use canonical face order"
            )
        expected_shaded = tuple(
            region.region_id for region in self.regions if region.shaded
        )
        if not self.regions[0].shaded:
            raise _validation_error(
                "blackboard_color_seed",
                "the least canonical face must use the deterministic shaded color",
            )
        if self.shaded_region_ids != expected_shaded:
            raise _validation_error(
                "blackboard_shaded_axis", "shaded vertices must retain region order"
            )
        if tuple(edge.crossing_id for edge in self.edges) != tuple(
            crossing.crossing_id for crossing in self.diagram.crossings
        ):
            raise _validation_error(
                "blackboard_crossing_axis", "edges must cover crossings in source order"
            )
        shaded = set(self.shaded_region_ids)
        shaded_by_region = {region.region_id: region.shaded for region in self.regions}
        region_of_dart = {
            dart: region.region_id
            for region in self.regions
            for dart in region.boundary_darts
        }
        for arc in self.diagram.arcs:
            if (
                shaded_by_region[region_of_dart[arc.tail]]
                == shaded_by_region[region_of_dart[arc.head]]
            ):
                raise _validation_error(
                    "blackboard_checkerboard_adjacency",
                    "every source projection arc must separate opposite checkerboard colors",
                )
        for edge, crossing in zip(self.edges, self.diagram.crossings, strict=True):
            corner_regions = tuple(region_of_dart[dart] for dart in crossing.half_edges)
            shaded_corners = tuple(
                index
                for index, region_id in enumerate(corner_regions)
                if region_id in shaded
            )
            if (
                len(shaded_corners) != 2
                or (shaded_corners[0] - shaded_corners[1]) % 2
                or shaded_corners != (edge.first_corner_index, edge.second_corner_index)
                or (
                    corner_regions[shaded_corners[0]],
                    corner_regions[shaded_corners[1]],
                )
                != (edge.first_region_id, edge.second_region_id)
            ):
                raise _validation_error(
                    "blackboard_edge_endpoints",
                    "crossing edge endpoints must match its opposite shaded source corners",
                )
            expected_tait_sign = (
                1 if set(shaded_corners) == set(crossing.over_pair) else -1
            )
            if edge.tait_sign != expected_tait_sign:
                raise _validation_error(
                    "blackboard_edge_sign", "Tait sign must match the source crossing"
                )
        return self


class BlackboardGraphRequest(StrictModel):
    diagram: OrientedLinkDiagram = Field(
        description=(
            "A classical oriented diagram with a connected nonempty crossing "
            "projection and at most 64 crossings; crossing-free loops are not "
            "part of a Tait graph."
        )
    )


class GoeritzDataRequest(StrictModel):
    diagram: OrientedLinkDiagram


class GoeritzDataResult(StrictModel):
    """A reduced Goeritz matrix derived from one typed checkerboard graph."""

    blackboard_graph: LinkBlackboardGraph
    deleted_region_id: LinkLabel
    reduced_matrix: IntegerMatrix
    absolute_determinant: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_goeritz_axes(self) -> Self:
        if self.deleted_region_id not in self.blackboard_graph.shaded_region_ids:
            raise _validation_error(
                "goeritz_deleted_region",
                "deleted region must belong to the shaded region axis",
            )
        expected_order = len(self.blackboard_graph.shaded_region_ids) - 1
        if (
            self.reduced_matrix.row_count != expected_order
            or self.reduced_matrix.column_count != expected_order
        ):
            raise _validation_error(
                "goeritz_matrix_shape",
                "reduced Goeritz matrix order must be shaded region count minus one",
            )
        return self


class LinkDeterminantResult(StrictModel):
    """The nonnegative knot determinant with its Alexander source value."""

    alexander: AlexanderPolynomialResult
    evaluation_at_minus_one: int
    determinant: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_absolute_evaluation(self) -> Self:
        if self.determinant != abs(self.evaluation_at_minus_one):
            raise _validation_error(
                "determinant_absolute_value",
                "knot determinant must be the absolute Alexander evaluation at -1",
            )
        return self


class LinkDeterminantRequest(StrictModel):
    """Compute the knot determinant from a bounded knot diagram."""

    diagram: OrientedLinkDiagram


class BraidLetter(StrictModel):
    """One signed Artin generator ``sigma_i^(+/-1)``."""

    generator: StrictInt = Field(ge=1, le=MAX_BRAID_STRANDS - 1)
    exponent: Literal[-1, 1]


class BraidWord(StrictModel):
    """A finite presentation word in the standard braid group ``B_n``."""

    strand_count: StrictInt = Field(ge=1, le=MAX_BRAID_STRANDS)
    letters: tuple[BraidLetter, ...] = Field(
        default=(), max_length=MAX_BRAID_WORD_LENGTH
    )

    @model_validator(mode="after")
    def require_generator_axis(self) -> Self:
        if any(letter.generator >= self.strand_count for letter in self.letters):
            raise _validation_error(
                "braid_generator_out_of_range",
                "every Artin generator index must satisfy 1 <= i < strand_count",
            )
        return self


class BraidWordRequest(StrictModel):
    word: BraidWord


class BraidProductRequest(StrictModel):
    """Multiply two bounded presentation words in one fixed braid group."""

    left: BraidWord
    right: BraidWord


class BraidPermutationResult(StrictModel):
    """The strand permutation and closure-cycle partition of a braid word."""

    word: BraidWord
    permutation: tuple[int, ...]
    cycles: tuple[tuple[int, ...], ...]
    closure_component_count: StrictInt = Field(ge=1, le=MAX_BRAID_STRANDS)
    exponent_sum: int

    @model_validator(mode="after")
    def require_structural_permutation(self) -> Self:
        size = self.word.strand_count
        if len(self.permutation) != size or sorted(self.permutation) != list(
            range(size)
        ):
            raise _validation_error(
                "braid_permutation_axis",
                "strand permutation must be total on the retained strand axis",
            )
        covered = tuple(sorted(item for cycle in self.cycles for item in cycle))
        if covered != tuple(range(size)) or self.closure_component_count != len(
            self.cycles
        ):
            raise _validation_error(
                "braid_cycle_partition",
                "closure cycles must partition every strand exactly once",
            )
        return self


class BraidClosureResult(StrictModel):
    """A source-bound standard closure as one canonical oriented diagram."""

    word: BraidWord
    permutation: BraidPermutationResult
    diagram: OrientedLinkDiagram

    @model_validator(mode="after")
    def require_structural_source_binding(self) -> Self:
        if self.permutation.word != self.word:
            raise _validation_error(
                "braid_closure_permutation_source",
                "closure permutation must bind the retained braid word",
            )
        if len(self.diagram.crossings) != len(self.word.letters):
            raise _validation_error(
                "braid_closure_crossing_count",
                "standard closure must retain one crossing per braid letter",
            )
        return self


class SeifertCircle(StrictModel):
    circle_id: LinkLabel
    darts: tuple[LinkLabel, ...] = Field(default=())


class SeifertCircleResult(StrictModel):
    """Canonical oriented smoothings and surface Euler data for one knot."""

    diagram: OrientedLinkDiagram
    circles: tuple[SeifertCircle, ...] = Field(min_length=1)
    band_crossing_ids: tuple[LinkLabel, ...]
    euler_characteristic: int
    genus: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_surface_axes(self) -> Self:
        if self.band_crossing_ids != tuple(
            crossing.crossing_id for crossing in self.diagram.crossings
        ):
            raise _validation_error(
                "seifert_band_axis",
                "Seifert bands must retain the complete source crossing axis",
            )
        if self.euler_characteristic != len(self.circles) - len(self.band_crossing_ids):
            raise _validation_error(
                "seifert_euler_characteristic",
                "surface Euler characteristic must equal disks minus bands",
            )
        if 2 * self.genus != 1 - self.euler_characteristic:
            raise _validation_error(
                "seifert_genus",
                "one-boundary-component surface genus must satisfy chi = 1 - 2g",
            )
        return self


class SeifertCircleRequest(StrictModel):
    diagram: OrientedLinkDiagram


class WirtingerPresentationRequest(StrictModel):
    diagram: OrientedLinkDiagram


class WirtingerArc(StrictModel):
    """One canonical Wirtinger generator and its diagram half-edge class."""

    generator_id: LinkLabel
    darts: tuple[LinkLabel, ...] = Field(default=())


class WirtingerCrossingRelator(StrictModel):
    crossing_id: LinkLabel
    over_generator: StrictInt = Field(ge=0, le=MAX_WIRTINGER_GENERATORS - 1)
    under_incoming_generator: StrictInt = Field(ge=0, le=MAX_WIRTINGER_GENERATORS - 1)
    under_outgoing_generator: StrictInt = Field(ge=0, le=MAX_WIRTINGER_GENERATORS - 1)
    word: FiniteGroupWord


class WirtingerPresentationResult(StrictModel):
    """A finite Wirtinger presentation with complete source transport."""

    diagram: OrientedLinkDiagram
    arcs: tuple[WirtingerArc, ...] = Field(
        min_length=1, max_length=MAX_WIRTINGER_GENERATORS
    )
    crossing_relators: tuple[WirtingerCrossingRelator, ...] = Field(
        max_length=MAX_LINK_CROSSINGS
    )
    presentation: FiniteGroupPresentation

    @model_validator(mode="after")
    def require_structural_presentation_binding(self) -> Self:
        generator_ids = tuple(arc.generator_id for arc in self.arcs)
        if self.presentation.generators != generator_ids:
            raise _validation_error(
                "wirtinger_generator_axis",
                "presentation generators must equal the retained Wirtinger arc axis",
            )
        if len(self.crossing_relators) != len(self.diagram.crossings) or tuple(
            row.crossing_id for row in self.crossing_relators
        ) != tuple(crossing.crossing_id for crossing in self.diagram.crossings):
            raise _validation_error(
                "wirtinger_crossing_axis",
                "Wirtinger relators must cover the source crossing axis in order",
            )
        if (
            tuple(row.word for row in self.crossing_relators)
            != self.presentation.relators
        ):
            raise _validation_error(
                "wirtinger_relator_binding",
                "crossing relators must equal the retained presentation relators",
            )
        return self


__all__ = [
    "MAX_BRAID_STRANDS",
    "MAX_BRAID_WORD_LENGTH",
    "MAX_WIRTINGER_GENERATORS",
    "AlexanderPolynomialRequest",
    "AlexanderPolynomialResult",
    "BlackboardGraphRequest",
    "BraidClosureResult",
    "BraidLetter",
    "BraidPermutationResult",
    "BraidWord",
    "BraidWordRequest",
    "CheckerboardRegion",
    "GoeritzDataRequest",
    "GoeritzDataResult",
    "LinkBlackboardEdge",
    "LinkBlackboardGraph",
    "LinkDeterminantRequest",
    "LinkDeterminantResult",
    "SeifertCircle",
    "SeifertCircleRequest",
    "SeifertCircleResult",
    "WirtingerArc",
    "WirtingerCrossingRelator",
    "WirtingerPresentationRequest",
    "WirtingerPresentationResult",
]
