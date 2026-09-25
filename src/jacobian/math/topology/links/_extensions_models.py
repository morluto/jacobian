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
