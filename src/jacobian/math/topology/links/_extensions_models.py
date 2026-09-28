"""Typed braid and Wirtinger contracts for classical link diagrams."""

from __future__ import annotations

from itertools import pairwise
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
    MAX_LINK_LABEL_LENGTH,
    LinkComponentsResult,
    LinkLabel,
    OrientedLinkDiagram,
)

MAX_BRAID_STRANDS = 32
MAX_BRAID_WORD_LENGTH = MAX_LINK_CROSSINGS
MAX_WIRTINGER_GENERATORS = 64
MAX_STATE_CIRCLE_CROSSINGS = MAX_LINK_CROSSINGS
MAX_CONWAY_CENTERED_DEGREE = 64
MAX_CONWAY_COEFFICIENT_DIGITS = 4_096

# Output bounds are materialization CELL counts derived from the link-diagram
# domain maxima (crossings, arcs, free loops, darts, and label characters), not
# transport-byte measures. Each crossing carries one crossing_id label and four
# half-edge dart labels, each at most MAX_LINK_LABEL_LENGTH characters.
_MAX_LINK_DIAGRAM_LABEL_CELLS = MAX_LINK_CROSSINGS * 5 * MAX_LINK_LABEL_LENGTH
_MAX_LINK_DIAGRAM_STRUCTURAL_CELLS = (
    MAX_LINK_CROSSINGS  # crossings
    + 2 * MAX_LINK_CROSSINGS  # arcs
    + MAX_LINK_CROSSINGS  # free loops
    + 4 * MAX_LINK_CROSSINGS  # darts
)
MAX_STATE_CIRCLE_OUTPUT_CELLS = (
    _MAX_LINK_DIAGRAM_LABEL_CELLS
    + _MAX_LINK_DIAGRAM_STRUCTURAL_CELLS
    + 20 * (4 * MAX_STATE_CIRCLE_CROSSINGS)
    + 8 * MAX_STATE_CIRCLE_CROSSINGS
    + 16 * MAX_STATE_CIRCLE_CROSSINGS
    + 128
)
MAX_CONWAY_OUTPUT_CELLS = (
    (MAX_CONWAY_CENTERED_DEGREE + 1) * (2 * MAX_CONWAY_COEFFICIENT_DIGITS + 128)
    + _MAX_LINK_DIAGRAM_LABEL_CELLS
    + _MAX_LINK_DIAGRAM_STRUCTURAL_CELLS
    + (2 * MAX_LINK_CROSSINGS + 1)  # retained Alexander polynomial terms
    + 256
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"link_diagram.{reason}", message)


MAX_LINK_DISJOINT_UNION_OUTPUT_BYTES = 8 * 1024 * 1024


class LinkDisjointUnionRequest(StrictModel):
    """A nonempty finite family of diagrams whose total size is bounded."""

    diagrams: tuple[OrientedLinkDiagram, ...] = Field(min_length=1, max_length=64)


class LinkDisjointUnionArcMap(StrictModel):
    source_index: StrictInt = Field(ge=0, le=63)
    source_tail: LinkLabel
    source_head: LinkLabel
    target_tail: LinkLabel
    target_head: LinkLabel


class LinkDisjointUnionCrossingMap(StrictModel):
    source_index: StrictInt = Field(ge=0, le=63)
    source_crossing_id: LinkLabel
    target_crossing_id: LinkLabel


class LinkDisjointUnionDartMap(StrictModel):
    source_index: StrictInt = Field(ge=0, le=63)
    source_dart_id: LinkLabel
    target_dart_id: LinkLabel


class LinkDisjointUnionFreeLoopMap(StrictModel):
    source_index: StrictInt = Field(ge=0, le=63)
    source_loop_index: StrictInt = Field(ge=0, le=63)
    target_loop_index: StrictInt = Field(ge=0, le=63)


class LinkDisjointUnionResult(StrictModel):
    """Tagged disjoint union with complete crossing, dart, arc, and loop transport."""

    sources: tuple[OrientedLinkDiagram, ...] = Field(min_length=1, max_length=64)
    diagram: OrientedLinkDiagram
    crossing_map: tuple[LinkDisjointUnionCrossingMap, ...] = Field(
        max_length=MAX_LINK_CROSSINGS
    )
    dart_map: tuple[LinkDisjointUnionDartMap, ...] = Field(
        max_length=4 * MAX_LINK_CROSSINGS
    )
    arc_map: tuple[LinkDisjointUnionArcMap, ...] = Field(
        max_length=2 * MAX_LINK_CROSSINGS
    )
    free_loop_map: tuple[LinkDisjointUnionFreeLoopMap, ...] = Field(
        max_length=MAX_LINK_CROSSINGS
    )

    def _require_unique_transport_keys(self) -> None:
        source_keys = (
            (
                self.crossing_map,
                lambda row: (row.source_index, row.source_crossing_id),
                lambda row: row.target_crossing_id,
            ),
            (
                self.dart_map,
                lambda row: (row.source_index, row.source_dart_id),
                lambda row: row.target_dart_id,
            ),
            (
                self.arc_map,
                lambda row: (row.source_index, row.source_tail, row.source_head),
                lambda row: (row.target_tail, row.target_head),
            ),
            (
                self.free_loop_map,
                lambda row: (row.source_index, row.source_loop_index),
                lambda row: row.target_loop_index,
            ),
        )
        for rows, source_key, target_key in source_keys:
            sources = tuple(source_key(row) for row in rows)
            targets = tuple(target_key(row) for row in rows)
            if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
                raise _validation_error(
                    "disjoint_union_transport_uniqueness",
                    "source and target transport keys must each be unique",
                )

    @model_validator(mode="after")
    def require_complete_source_transport(self) -> Self:
        crossing_count = sum(len(source.crossings) for source in self.sources)
        free_loop_count = sum(source.free_loops for source in self.sources)
        if crossing_count > MAX_LINK_CROSSINGS or free_loop_count > MAX_LINK_CROSSINGS:
            raise _validation_error(
                "disjoint_union_source_bound",
                "source family exceeds the bounded union size",
            )
        self._require_unique_transport_keys()

        crossings = {row.crossing_id: row for row in self.diagram.crossings}
        darts = {
            dart for crossing in self.diagram.crossings for dart in crossing.half_edges
        }
        arcs = {(arc.tail, arc.head) for arc in self.diagram.arcs}
        crossing_sources = {
            (i, crossing.crossing_id)
            for i, source in enumerate(self.sources)
            for crossing in source.crossings
        }
        dart_sources = {
            (i, dart)
            for i, source in enumerate(self.sources)
            for crossing in source.crossings
            for dart in crossing.half_edges
        }
        arc_sources = {
            (i, arc.tail, arc.head)
            for i, source in enumerate(self.sources)
            for arc in source.arcs
        }
        loop_sources = {
            (i, loop)
            for i, source in enumerate(self.sources)
            for loop in range(source.free_loops)
        }
        if {
            (r.source_index, r.source_crossing_id) for r in self.crossing_map
        } != crossing_sources or len(self.crossing_map) != len(crossing_sources):
            raise _validation_error(
                "disjoint_union_crossing_coverage",
                "crossing transport must cover every source crossing",
            )
        if {
            (r.source_index, r.source_dart_id) for r in self.dart_map
        } != dart_sources or len(self.dart_map) != len(dart_sources):
            raise _validation_error(
                "disjoint_union_dart_coverage",
                "dart transport must cover every source dart",
            )
        if {
            (r.source_index, r.source_tail, r.source_head) for r in self.arc_map
        } != arc_sources or len(self.arc_map) != len(arc_sources):
            raise _validation_error(
                "disjoint_union_arc_coverage",
                "arc transport must cover every source arc",
            )
        if {
            (r.source_index, r.source_loop_index) for r in self.free_loop_map
        } != loop_sources or len(self.free_loop_map) != len(loop_sources):
            raise _validation_error(
                "disjoint_union_loop_coverage",
                "loop transport must cover every source loop",
            )
        if (
            len(darts) != len(self.dart_map)
            or len(arcs) != len(self.arc_map)
            or len(crossings) != len(self.crossing_map)
        ):
            raise _validation_error(
                "disjoint_union_target_coverage",
                "transport targets must cover the target diagram",
            )
        dart_maps = {
            (r.source_index, r.source_dart_id): r.target_dart_id for r in self.dart_map
        }
        for crossing_row in self.crossing_map:
            source = self.sources[crossing_row.source_index]
            source_crossing = next(
                c
                for c in source.crossings
                if c.crossing_id == crossing_row.source_crossing_id
            )
            target = crossings.get(crossing_row.target_crossing_id)
            mapped_darts = tuple(
                dart_maps[(crossing_row.source_index, dart)]
                for dart in source_crossing.half_edges
            )
            if (
                target is None
                or target.half_edges != mapped_darts
                or (target.over_pair, target.under_pair, target.sign)
                != (
                    source_crossing.over_pair,
                    source_crossing.under_pair,
                    source_crossing.sign,
                )
            ):
                raise _validation_error(
                    "disjoint_union_crossing_binding",
                    "crossing transport must bind matching source metadata",
                )
        arc_rows = {
            (r.source_index, r.source_tail, r.source_head): r for r in self.arc_map
        }
        for i, source in enumerate(self.sources):
            for arc in source.arcs:
                row = arc_rows[(i, arc.tail, arc.head)]
                if (row.target_tail, row.target_head) != (
                    dart_maps[(i, arc.tail)],
                    dart_maps[(i, arc.head)],
                ) or (row.target_tail, row.target_head) not in arcs:
                    raise _validation_error(
                        "disjoint_union_arc_binding",
                        "arc transport must bind directed source and target arcs",
                    )
        expected_loop_targets = {
            (source_index, local_index): sum(
                source.free_loops for source in self.sources[:source_index]
            )
            + local_index
            for source_index, source in enumerate(self.sources)
            for local_index in range(source.free_loops)
        }
        if self.diagram.free_loops != free_loop_count or any(
            row.target_loop_index
            != expected_loop_targets[(row.source_index, row.source_loop_index)]
            for row in self.free_loop_map
        ):
            raise _validation_error(
                "disjoint_union_target_loops",
                "loop transport must preserve each source's cumulative offset",
            )
        return self


class LinkSignatureRequest(StrictModel):
    """Compute the oriented link signature from one bounded diagram."""

    diagram: OrientedLinkDiagram


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
    """Conway polynomial with the Laurent unit relating it to its Alexander value."""

    alexander: AlexanderPolynomialResult
    polynomial: RationalLaurentPolynomial
    alexander_unit_sign: Literal[-1, 1]
    alexander_unit_power: StrictInt
    normalization: Literal[
        "sign*t^power*Alexander(t)=Conway(t^(1/2)-t^(-1/2)); Alexander(1)=1"
    ] = "sign*t^power*Alexander(t)=Conway(t^(1/2)-t^(-1/2)); Alexander(1)=1"

    @model_validator(mode="after")
    def require_conway_polynomial_context(self) -> Self:
        if self.polynomial.variables != ("z",):
            raise _validation_error(
                "conway_polynomial_variable",
                "Conway polynomial must use the canonical variable z",
            )
        if abs(self.alexander_unit_power) > MAX_CONWAY_CENTERED_DEGREE:
            raise _validation_error(
                "conway_alexander_unit_power",
                "Alexander centering power exceeds the admitted Conway degree",
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
    circle_count: StrictInt = Field(ge=1, le=3 * MAX_LINK_CROSSINGS)

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
        expected_count = (
            len({circle.darts for circle in self.circles if circle.darts})
            + diagram.free_loops
        )
        if self.circle_count != expected_count or len(self.circles) != expected_count:
            raise _validation_error(
                "state_circle_count", "circle count must equal the retained circle axis"
            )
        if self.circles != tuple(sorted(self.circles, key=lambda circle: circle.darts)):
            raise _validation_error(
                "state_circle_order", "circles must use canonical lexicographic order"
            )
        return self


class GoeritzRegion(StrictModel):
    region_id: LinkLabel
    boundary_darts: tuple[LinkLabel, ...] = Field(min_length=1)
    shaded: bool


class GoeritzCrossingContribution(StrictModel):
    crossing_id: LinkLabel
    first_region_id: LinkLabel
    second_region_id: LinkLabel
    incidence: Literal[-1, 1]


class GoeritzDataRequest(StrictModel):
    diagram: OrientedLinkDiagram


class GoeritzDataResult(StrictModel):
    """A deterministic checkerboard shading and its reduced Goeritz matrix."""

    diagram: OrientedLinkDiagram
    regions: tuple[GoeritzRegion, ...] = Field(min_length=2)
    shaded_region_ids: tuple[LinkLabel, ...] = Field(min_length=1)
    crossing_contributions: tuple[GoeritzCrossingContribution, ...]
    deleted_region_id: LinkLabel
    reduced_matrix: IntegerMatrix
    absolute_determinant: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_goeritz_axes(self) -> Self:
        region_ids = tuple(region.region_id for region in self.regions)
        if len(set(region_ids)) != len(region_ids):
            raise _validation_error(
                "goeritz_region_ids", "checkerboard region IDs must be unique"
            )
        expected_shaded = tuple(
            region.region_id for region in self.regions if region.shaded
        )
        if self.shaded_region_ids != expected_shaded:
            raise _validation_error(
                "goeritz_shaded_axis",
                "shaded region axis must equal the retained checkerboard shading",
            )
        if self.deleted_region_id not in self.shaded_region_ids:
            raise _validation_error(
                "goeritz_deleted_region",
                "deleted region must belong to the shaded region axis",
            )
        expected_order = len(self.shaded_region_ids) - 1
        if (
            self.reduced_matrix.row_count != expected_order
            or self.reduced_matrix.column_count != expected_order
        ):
            raise _validation_error(
                "goeritz_matrix_shape",
                "reduced Goeritz matrix order must be shaded region count minus one",
            )
        if tuple(row.crossing_id for row in self.crossing_contributions) != tuple(
            crossing.crossing_id for crossing in self.diagram.crossings
        ):
            raise _validation_error(
                "goeritz_crossing_axis",
                "crossing contributions must retain the complete source axis",
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


class BraidArtinActionResult(StrictModel):
    """Exact free-group images of the Artin automorphism of a braid."""

    word: BraidWord
    generator_images: tuple[FiniteGroupWord, ...] = Field(
        min_length=1, max_length=MAX_BRAID_STRANDS
    )

    @model_validator(mode="after")
    def require_complete_free_group_endomorphism(self) -> Self:
        if len(self.generator_images) != self.word.strand_count:
            raise _validation_error(
                "artin_action_generator_axis",
                "the action must provide one image for each braid strand generator",
            )
        for image in self.generator_images:
            if any(
                letter.generator >= self.word.strand_count for letter in image.letters
            ):
                raise _validation_error(
                    "artin_action_generator_index",
                    "every image letter must name a generator on the retained axis",
                )
            if any(
                left.generator == right.generator and left.exponent == -right.exponent
                for left, right in pairwise(image.letters)
            ):
                raise _validation_error(
                    "artin_action_not_reduced",
                    "each free-group image must be freely reduced",
                )
        return self


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
    "MAX_LINK_CROSSINGS",
    "MAX_WIRTINGER_GENERATORS",
    "AlexanderPolynomialRequest",
    "AlexanderPolynomialResult",
    "BraidClosureResult",
    "BraidLetter",
    "BraidPermutationResult",
    "BraidWord",
    "BraidWordRequest",
    "GoeritzCrossingContribution",
    "GoeritzDataRequest",
    "GoeritzDataResult",
    "GoeritzRegion",
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
