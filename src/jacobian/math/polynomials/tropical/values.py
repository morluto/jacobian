"""Canonical exact values for bounded tropical algebra."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_TOTAL_FACES,
    PolytopalComplexClosureResult,
)
from jacobian.math.geometry.polytopes.values import (
    RationalHPolyhedron,
    RationalPolyhedronVPresentation,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import rational_rank

MAX_TROPICAL_SCALAR_DIGITS = 8_192
MAX_TROPICAL_VECTOR_DIMENSION = 128
MAX_TROPICAL_MATRIX_CELLS = 4_096
MAX_TROPICAL_POLYNOMIAL_TERMS = 512
MAX_TROPICAL_EXPONENT = 1_024
MAX_TROPICAL_ROOT_CROSSOVER_PAIRS = 65_536
MAX_TROPICAL_ROOT_RESULT_BYTES = 16 * 1024 * 1024
MAX_TROPICAL_ROOT_DIGITS = 16_384
MAX_TROPICAL_NEWTON_RESULT_BYTES = 16 * 1024 * 1024
MAX_TROPICAL_SUBDIVISION_TERMS = 10
MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS = 32
MAX_TROPICAL_SUBDIVISION_RESULT_BYTES = 4 * 1024 * 1024
MAX_TROPICAL_ESSENTIAL_VARIABLES = 4
MAX_TROPICAL_ESSENTIAL_TERMS = 64
MAX_TROPICAL_ESSENTIAL_FACES = 10_000
MAX_TROPICAL_ESSENTIAL_WORK = 2_000_000
MAX_TROPICAL_ESSENTIAL_RESULT_BYTES = 10 * 1024 * 1024
MAX_TROPICAL_HYPERSURFACE_CELLS = 40
MAX_TROPICAL_HYPERSURFACE_RESULT_BYTES = 10 * 1024 * 1024
MAX_TROPICAL_ACTIVE_TERM_WORK = 250_000_000
MAX_TROPICAL_ACTIVE_RESULT_BYTES = 12 * 1024 * 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tropical.{reason}", message)


class TropicalSemiring(StrictModel):
    convention: Literal["MIN_PLUS", "MAX_PLUS"]
    base: Literal["ZZ", "QQ"]


class TropicalScalar(StrictModel):
    semiring: TropicalSemiring
    kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"]
    value: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_licensed_scalar(self) -> Self:
        if self.kind == "FINITE":
            if self.value is None:
                raise _validation_error(
                    "finite_missing_value", "a finite scalar must carry its value"
                )
            if self.semiring.base == "ZZ" and self.value.den != 1:
                raise _validation_error(
                    "nonintegral_integer_scalar",
                    "a ZZ tropical scalar must be integral",
                )
        else:
            if self.value is not None:
                raise _validation_error(
                    "infinite_carries_value", "an infinite scalar carries no value"
                )
            if (self.kind == "POSITIVE_INFINITY") != (
                self.semiring.convention == "MIN_PLUS"
            ):
                raise _validation_error(
                    "unlicensed_infinity",
                    "only the semiring-licensed infinity is an element",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        semiring: TropicalSemiring,
        kind: Literal["FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"],
        value: CanonicalRational | None,
    ) -> Self:
        return cls.model_construct(semiring=semiring, kind=kind, value=value)


def require_scalar_budget(scalar: TropicalScalar) -> None:
    if scalar.value is not None:
        try:
            require_bounded_rational(
                scalar.value,
                max_digits=MAX_TROPICAL_SCALAR_DIGITS,
                label="tropical scalar",
            )
        except ValueError as error:
            raise _validation_error("scalar_budget", str(error)) from error


class TropicalVector(StrictModel):
    semiring: TropicalSemiring
    axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    entries: tuple[TropicalScalar, ...] = Field(
        max_length=MAX_TROPICAL_VECTOR_DIMENSION
    )

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if len(set(self.axis)) != len(self.axis) or len(self.axis) != len(self.entries):
            raise _validation_error(
                "vector_shape",
                "vector axis and entries must have the same unique labels",
            )
        if any(entry.semiring != self.semiring for entry in self.entries):
            raise _validation_error(
                "vector_semiring", "every vector entry must carry the vector semiring"
            )
        return self


class TropicalMatrix(StrictModel):
    semiring: TropicalSemiring
    row_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    column_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_TROPICAL_VECTOR_DIMENSION
    )
    entries: tuple[tuple[TropicalScalar, ...], ...]

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if len(set(self.row_axis)) != len(self.row_axis) or len(
            set(self.column_axis)
        ) != len(self.column_axis):
            raise _validation_error(
                "matrix_axis", "matrix axes must have unique labels"
            )
        if len(self.entries) != len(self.row_axis) or any(
            len(row) != len(self.column_axis) for row in self.entries
        ):
            raise _validation_error(
                "matrix_shape", "matrix entries must match row and column axes"
            )
        if any(
            entry.semiring != self.semiring for row in self.entries for entry in row
        ):
            raise _validation_error(
                "matrix_semiring", "every matrix entry must carry the matrix semiring"
            )
        if len(self.row_axis) * len(self.column_axis) > MAX_TROPICAL_MATRIX_CELLS:
            raise _validation_error(
                "matrix_budget", "tropical matrix has too many cells"
            )
        return self


class TropicalPolynomialTerm(StrictModel):
    exponents: tuple[int, ...]
    coefficient: TropicalScalar

    @model_validator(mode="after")
    def require_exponents(self) -> Self:
        if any(value < 0 or value > MAX_TROPICAL_EXPONENT for value in self.exponents):
            raise _validation_error(
                "exponent_bound", "exponents must be nonnegative and bounded"
            )
        return self


class TropicalPolynomial(StrictModel):
    semiring: TropicalSemiring
    variables: tuple[OpaqueLabel, ...] = Field(max_length=MAX_TROPICAL_VECTOR_DIMENSION)
    terms: tuple[TropicalPolynomialTerm, ...] = Field(
        max_length=MAX_TROPICAL_POLYNOMIAL_TERMS
    )

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "polynomial_axis", "polynomial variables must be unique"
            )
        exponents = tuple(term.exponents for term in self.terms)
        if any(len(exp) != len(self.variables) for exp in exponents):
            raise _validation_error(
                "polynomial_shape", "term exponents must match variables"
            )
        if any(term.coefficient.semiring != self.semiring for term in self.terms):
            raise _validation_error(
                "polynomial_semiring", "terms must carry the polynomial semiring"
            )
        if exponents != tuple(sorted(exponents)) or len(set(exponents)) != len(
            exponents
        ):
            raise _validation_error(
                "polynomial_terms",
                "polynomial terms must be sorted with unique exponents",
            )
        if any(term.coefficient.kind != "FINITE" for term in self.terms):
            raise _validation_error(
                "polynomial_infinity",
                "infinite coefficients are omitted from canonical polynomials",
            )
        return self


class TropicalRootInterval(StrictModel):
    """One maximal open interval with a fixed active affine term."""

    lower: CanonicalRational | None
    upper: CanonicalRational | None
    active_exponent: int
    slope: int


class TropicalRootBreakpoint(StrictModel):
    """A finite corner and all terms tied at its exact location."""

    value: CanonicalRational
    multiplicity: int
    left_exponent: int
    right_exponent: int
    left_slope: int
    right_slope: int
    active_exponents: tuple[int, ...]


class TropicalUnivariateRootProfile(StrictModel):
    """Complete exact piecewise-linear profile for one tropical variable."""

    source: TropicalPolynomial
    kind: Literal["ZERO_POLYNOMIAL", "FINITE_PROFILE"]
    intervals: tuple[TropicalRootInterval, ...]
    roots: tuple[TropicalRootBreakpoint, ...]

    @model_validator(mode="after")
    def require_canonical_profile(self) -> Self:
        if len(self.source.variables) != 1:
            raise _validation_error(
                "root_profile_axis", "root profile source must be univariate"
            )
        if self.kind == "ZERO_POLYNOMIAL":
            if self.source.terms or self.intervals or self.roots:
                raise _validation_error(
                    "root_profile_zero", "zero polynomial has no finite affine profile"
                )
            return self
        if not self.source.terms or len(self.intervals) != len(self.roots) + 1:
            raise _validation_error(
                "root_profile_shape", "finite profile needs one interval per root gap"
            )
        if self.intervals[0].lower is not None or self.intervals[-1].upper is not None:
            raise _validation_error(
                "root_profile_ends", "profile must include both unbounded end intervals"
            )
        for index, root in enumerate(self.roots):
            left, right = self.intervals[index], self.intervals[index + 1]
            if (
                root.multiplicity != abs(root.right_slope - root.left_slope)
                or root.left_slope != left.slope
                or root.right_slope != right.slope
                or root.left_exponent != left.active_exponent
                or root.right_exponent != right.active_exponent
                or root.value != left.upper
                or root.value != right.lower
                or tuple(sorted(set(root.active_exponents))) != root.active_exponents
                or root.left_exponent not in root.active_exponents
                or root.right_exponent not in root.active_exponents
            ):
                raise _validation_error(
                    "root_profile_breakpoint",
                    "root breakpoint disagrees with intervals",
                )
        if any(
            left.value.as_fraction() >= right.value.as_fraction()
            for left, right in zip(self.roots, self.roots[1:], strict=False)
        ):
            raise _validation_error(
                "root_profile_order", "roots must be strictly increasing"
            )
        return self


class TropicalNewtonPolygonVertex(StrictModel):
    """A hull vertex with its source-term position and exact lifted point."""

    source_term_index: int = Field(ge=0, le=MAX_TROPICAL_POLYNOMIAL_TERMS - 1)
    exponent: int = Field(ge=0, le=MAX_TROPICAL_EXPONENT)
    coefficient: TropicalScalar


class TropicalNewtonPolygonEdge(StrictModel):
    """A maximal lower/upper face, including all collinear source terms."""

    left_vertex_index: int = Field(ge=0, le=MAX_TROPICAL_POLYNOMIAL_TERMS - 1)
    right_vertex_index: int = Field(ge=1, le=MAX_TROPICAL_POLYNOMIAL_TERMS - 1)
    source_term_indices: tuple[int, ...] = Field(
        min_length=2, max_length=MAX_TROPICAL_POLYNOMIAL_TERMS
    )
    slope: CanonicalRational
    tropical_root: CanonicalRational
    multiplicity: int = Field(ge=1, le=MAX_TROPICAL_EXPONENT)


class TropicalNewtonPolygonProfile(StrictModel):
    """Exact one-variable coefficient hull bound to its formal source."""

    source: TropicalPolynomial
    hull_vertices: tuple[TropicalNewtonPolygonVertex, ...] = Field(
        max_length=MAX_TROPICAL_POLYNOMIAL_TERMS
    )
    edges: tuple[TropicalNewtonPolygonEdge, ...] = Field(
        max_length=MAX_TROPICAL_POLYNOMIAL_TERMS - 1
    )

    @model_validator(mode="after")
    def require_profile_shape(self) -> Self:
        if len(self.source.variables) != 1 or len(self.edges) != max(
            0, len(self.hull_vertices) - 1
        ):
            raise _validation_error(
                "newton_profile_shape",
                "Newton polygon profile requires one variable and consecutive hull edges",
            )
        for index, vertex in enumerate(self.hull_vertices):
            if (
                vertex.source_term_index >= len(self.source.terms)
                or vertex.exponent
                != self.source.terms[vertex.source_term_index].exponents[0]
                or vertex.coefficient
                != self.source.terms[vertex.source_term_index].coefficient
                or (index and vertex.exponent <= self.hull_vertices[index - 1].exponent)
            ):
                raise _validation_error(
                    "newton_vertex_source",
                    "hull vertices must preserve ordered source terms",
                )
        for index, edge in enumerate(self.edges):
            if (
                edge.left_vertex_index != index
                or edge.right_vertex_index != index + 1
                or edge.multiplicity
                != self.hull_vertices[index + 1].exponent
                - self.hull_vertices[index].exponent
                or tuple(sorted(set(edge.source_term_indices)))
                != edge.source_term_indices
                or edge.source_term_indices[0]
                != self.hull_vertices[index].source_term_index
                or edge.source_term_indices[-1]
                != self.hull_vertices[index + 1].source_term_index
                or any(
                    term_index >= len(self.source.terms)
                    for term_index in edge.source_term_indices
                )
            ):
                raise _validation_error(
                    "newton_edge_shape",
                    "Newton edges must retain ordered source provenance",
                )
        return self


class TropicalLiftedSubdivisionFace(StrictModel):
    """One source-bound lower/upper face of the lifted coefficient hull."""

    lifted_face_index: int = Field(ge=0, le=15)
    source_hull_facet_index: int | None = Field(default=None, ge=0, le=255)
    normal: tuple[CanonicalRational, CanonicalRational, CanonicalRational]
    offset: CanonicalRational
    source_term_indices: tuple[int, ...] = Field(
        min_length=3, max_length=MAX_TROPICAL_SUBDIVISION_TERMS
    )
    subdivision_cell_id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_source_order(self) -> Self:
        if tuple(sorted(set(self.source_term_indices))) != self.source_term_indices:
            raise _validation_error(
                "lifted_face_source_order",
                "lifted-face source term indices must be sorted and unique",
            )
        return self


class TropicalSubdivisionFaceSupport(StrictModel):
    """Source monomials and lifted facets dual to one projected complex face."""

    face_id: str = Field(min_length=1, max_length=64)
    dimension: int = Field(ge=-1, le=2)
    source_term_indices: tuple[int, ...] = Field(
        max_length=MAX_TROPICAL_SUBDIVISION_TERMS
    )
    lifted_face_indices: tuple[int, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def require_sorted_provenance(self) -> Self:
        if (
            tuple(sorted(set(self.source_term_indices))) != self.source_term_indices
            or tuple(sorted(set(self.lifted_face_indices))) != self.lifted_face_indices
        ):
            raise _validation_error(
                "subdivision_face_provenance",
                "subdivision provenance indices must be sorted and unique",
            )
        return self


class TropicalRegularSubdivision(StrictModel):
    """A source-bound bivariate regular subdivision in its exact cell complex."""

    source: TropicalPolynomial
    cell_complex: PolytopalComplexClosureResult
    lifted_faces: tuple[TropicalLiftedSubdivisionFace, ...] = Field(
        min_length=1, max_length=16
    )
    face_supports: tuple[TropicalSubdivisionFaceSupport, ...] = Field(
        min_length=2, max_length=MAX_COMPLEX_TOTAL_FACES
    )

    @model_validator(mode="after")
    def require_source_bound_complex(self) -> Self:
        if (
            len(self.source.variables) != 2
            or self.cell_complex.dimension != 2
            or self.cell_complex.space.axes != self.source.variables
            or len(self.lifted_faces) != len(self.cell_complex.maximal_cells)
            or len(self.face_supports) != len(self.cell_complex.faces)
            or tuple(face.lifted_face_index for face in self.lifted_faces)
            != tuple(range(len(self.lifted_faces)))
        ):
            raise _validation_error(
                "subdivision_source_shape",
                "regular subdivision must retain its bivariate source and complete planar complex",
            )
        maximal_ids = {cell.cell_id for cell in self.cell_complex.maximal_cells}
        if any(
            face.subdivision_cell_id not in maximal_ids for face in self.lifted_faces
        ):
            raise _validation_error(
                "subdivision_cell_binding",
                "every lifted face must map to a maximal cell of the returned complex",
            )
        if any(
            term_index >= len(self.source.terms)
            for lifted_face in self.lifted_faces
            for term_index in lifted_face.source_term_indices
        ):
            raise _validation_error(
                "lifted_face_term_index",
                "lifted-face source indices must refer to the retained polynomial",
            )
        if any(
            face.normal[2].num >= 0
            if self.source.semiring.convention == "MIN_PLUS"
            else face.normal[2].num <= 0
            for face in self.lifted_faces
        ):
            raise _validation_error(
                "subdivision_orientation",
                "lifted face orientation must match the min-plus lower or max-plus upper convention",
            )
        for support, face in zip(
            self.face_supports, self.cell_complex.faces, strict=True
        ):
            if support.face_id != face.face_id or support.dimension != face.dimension:
                raise _validation_error(
                    "subdivision_face_binding",
                    "term supports must match the complete canonical face order",
                )
            if any(
                index >= len(self.source.terms) for index in support.source_term_indices
            ):
                raise _validation_error(
                    "subdivision_term_index",
                    "term support indices must refer to the retained source polynomial",
                )
            if any(
                index >= len(self.lifted_faces) for index in support.lifted_face_indices
            ):
                raise _validation_error(
                    "subdivision_lifted_face_index",
                    "dual lifted-face indices must refer to the returned lifted faces",
                )
        return self


class TropicalEssentialLiftedFace(StrictModel):
    """One finite-normal lifted hull face, bound to source terms."""

    face_index: int | None = Field(
        default=None, ge=0, le=MAX_TROPICAL_ESSENTIAL_FACES - 1
    )
    dimension: int = Field(ge=0, le=6)
    normal: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=7)
    offset: CanonicalRational
    source_term_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_TROPICAL_ESSENTIAL_TERMS
    )


class TropicalEssentialHullFace(StrictModel):
    """Source incidence for a complete finite-normal lower/upper face."""

    face_index: int = Field(ge=0, le=MAX_TROPICAL_ESSENTIAL_FACES - 1)
    dimension: int = Field(ge=0, le=6)
    source_term_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_TROPICAL_ESSENTIAL_TERMS
    )
    maximal_finite_face_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_TROPICAL_ESSENTIAL_FACES
    )


class TropicalPolynomialEssentialPart(StrictModel):
    """Tie-inclusive attained support and its finite-normal hull incidence."""

    source: TropicalPolynomial
    polynomial: TropicalPolynomial
    essential_term_indices: tuple[int, ...]
    inessential_term_indices: tuple[int, ...]
    variable_indices: tuple[int, ...] = Field(max_length=6)
    lifted_affine_dimension: int = Field(ge=0, le=7)
    affine_equalities: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=7,
        description=(
            "Rows of exact lifted affine equations: variable coefficients, the "
            "coefficient-height component, then the right-hand side."
        ),
    )
    hull_facets: tuple[TropicalEssentialLiftedFace, ...] = Field(
        max_length=MAX_TROPICAL_ESSENTIAL_FACES
    )
    finite_faces: tuple[TropicalEssentialLiftedFace, ...] = Field(
        max_length=MAX_TROPICAL_ESSENTIAL_FACES
    )
    face_incidence: tuple[TropicalEssentialHullFace, ...] = Field(
        max_length=MAX_TROPICAL_ESSENTIAL_FACES
    )

    @model_validator(mode="after")
    def require_source_partition(self) -> Self:
        all_indices = tuple(range(len(self.source.terms)))
        if (
            self.polynomial.semiring != self.source.semiring
            or self.polynomial.variables != self.source.variables
            or self.variable_indices != tuple(range(len(self.source.variables)))
            or self.lifted_affine_dimension > len(self.source.variables) + 1
            or any(
                len(face.normal) != len(self.source.variables) + 1
                for face in (*self.hull_facets, *self.finite_faces)
            )
            or any(
                len(row) != len(self.source.variables) + 2
                for row in self.affine_equalities
            )
            or tuple(
                sorted((*self.essential_term_indices, *self.inessential_term_indices))
            )
            != all_indices
            or set(self.essential_term_indices) & set(self.inessential_term_indices)
            or tuple(sorted(self.essential_term_indices)) != self.essential_term_indices
            or tuple(sorted(self.inessential_term_indices))
            != self.inessential_term_indices
            or tuple(face.face_index for face in self.hull_facets)
            != tuple(range(len(self.hull_facets)))
            or tuple(face.face_index for face in self.face_incidence)
            != tuple(range(len(self.face_incidence)))
        ):
            raise _validation_error(
                "essential_part_source_shape",
                "essential result must preserve its polynomial axes, partition, and indexed hull faces",
            )
        expected_terms = tuple(
            self.source.terms[index] for index in self.essential_term_indices
        )
        if self.polynomial.terms != expected_terms:
            raise _validation_error(
                "essential_part_polynomial",
                "returned polynomial must contain exactly the attained source terms in source order",
            )
        finite_support = {
            index for face in self.finite_faces for index in face.source_term_indices
        }
        if finite_support != set(self.essential_term_indices):
            raise _validation_error(
                "essential_part_face_coverage",
                "finite-normal face incidence must equal the attained source support",
            )
        lifted_points = tuple(
            (
                *term.exponents,
                term.coefficient.value.as_fraction(),
            )
            for term in self.source.terms
            if term.coefficient.value is not None
        )
        if any(
            not any(value.num for value in row[:-1])
            or any(
                sum(
                    coefficient.as_fraction() * coordinate
                    for coefficient, coordinate in zip(row[:-1], point, strict=True)
                )
                != row[-1].as_fraction()
                for point in lifted_points
            )
            for row in self.affine_equalities
        ):
            raise _validation_error(
                "essential_part_affine_equalities",
                "affine equalities must be nontrivial exact relations on every lifted source term",
            )
        if any(
            face.normal[-1].num >= 0
            if self.source.semiring.convention == "MIN_PLUS"
            else face.normal[-1].num <= 0
            for face in self.finite_faces
        ):
            raise _validation_error(
                "essential_part_face_orientation",
                "finite-face orientation must match the min-plus lower or max-plus upper convention",
            )
        if any(
            tuple(sorted(set(face.source_term_indices))) != face.source_term_indices
            or any(
                index < 0 or index >= len(self.source.terms)
                for index in face.source_term_indices
            )
            for face in (*self.hull_facets, *self.finite_faces)
        ):
            raise _validation_error(
                "essential_part_incidence",
                "lifted face incidence must use ordered source term indices",
            )
        if any(
            not face.maximal_finite_face_indices
            or any(
                index < 0 or index >= len(self.finite_faces)
                for index in face.maximal_finite_face_indices
            )
            or tuple(sorted(set(face.maximal_finite_face_indices)))
            != face.maximal_finite_face_indices
            or tuple(sorted(set(face.source_term_indices))) != face.source_term_indices
            or any(
                index < 0 or index >= len(self.source.terms)
                for index in face.source_term_indices
            )
            or any(
                face.dimension > self.finite_faces[index].dimension
                or not set(face.source_term_indices).issubset(
                    self.finite_faces[index].source_term_indices
                )
                for index in face.maximal_finite_face_indices
            )
            for face in self.face_incidence
        ):
            raise _validation_error(
                "essential_part_face_incidence",
                "each lifted source face must map to its finite-normal parent faces",
            )
        if any(
            term.coefficient.kind != "FINITE" or term.coefficient.value is None
            for term in self.source.terms
        ):
            raise _validation_error(
                "essential_part_source_height",
                "lifted hull incidence requires finite source coefficients",
            )
        points = tuple(
            (
                *(Fraction(exponent) for exponent in term.exponents),
                Fraction(term.coefficient.value.num, term.coefficient.value.den),
            )
            for term in self.source.terms
        )

        def support_dimension(indices: tuple[int, ...]) -> int:
            if len(indices) < 2:
                return 0
            origin = points[indices[0]]
            rows = [
                [points[index][axis] - origin[axis] for axis in range(len(origin))]
                for index in indices[1:]
            ]
            return rational_rank(rows, len(origin))

        def matches_support(face: TropicalEssentialLiftedFace) -> bool:
            normal = tuple(
                Fraction(component.num, component.den) for component in face.normal
            )
            offset = Fraction(face.offset.num, face.offset.den)
            values = tuple(
                sum((a * b for a, b in zip(normal, point, strict=True)), Fraction(0))
                for point in points
            )
            equality = tuple(i for i, value in enumerate(values) if value == offset)
            return (
                equality == face.source_term_indices
                and support_dimension(equality) == face.dimension
                and (
                    all(value <= offset for value in values)
                    or all(value >= offset for value in values)
                )
            )

        if (
            any(not matches_support(face) for face in self.hull_facets)
            or any(not matches_support(face) for face in self.finite_faces)
            or any(
                face.face_index is not None
                and (
                    face.face_index >= len(self.hull_facets)
                    or face != self.hull_facets[face.face_index]
                )
                for face in self.finite_faces
            )
            or any(
                support_dimension(face.source_term_indices) != face.dimension
                for face in self.face_incidence
            )
            or support_dimension(all_indices) != self.lifted_affine_dimension
        ):
            raise _validation_error(
                "essential_part_face_geometry",
                "lifted face equations, incidence, dimensions, and hull indices must match the source points",
            )
        return self


class TropicalHypersurfaceCell(StrictModel):
    """One exact corner cell bound to its dual subdivision face."""

    cell_id: str = Field(min_length=1, max_length=16)
    dimension: int = Field(ge=0, le=1)
    inequalities: RationalHPolyhedron
    generators: RationalPolyhedronVPresentation
    active_term_indices: tuple[int, ...] = Field(
        min_length=2, max_length=MAX_TROPICAL_SUBDIVISION_TERMS
    )
    dual_face_id: str = Field(min_length=1, max_length=64)
    dual_lifted_face_indices: tuple[int, ...] = Field(max_length=16)
    incident_cell_ids: tuple[str, ...] = Field(
        max_length=MAX_TROPICAL_HYPERSURFACE_CELLS
    )
    weight: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_exact_cell_shape(self) -> Self:
        if (
            self.inequalities.space.axes != self.generators.space.axes
            or self.generators.empty
            or self.generators.affine_dimension != self.dimension
            or (self.dimension == 0) != (self.weight is None)
            or tuple(sorted(set(self.active_term_indices))) != self.active_term_indices
            or tuple(sorted(set(self.dual_lifted_face_indices)))
            != self.dual_lifted_face_indices
            or tuple(sorted(set(self.incident_cell_ids))) != self.incident_cell_ids
            or self.cell_id in self.incident_cell_ids
        ):
            raise _validation_error(
                "hypersurface_cell_shape",
                "corner-cell geometry, dimension, weight, and provenance must agree",
            )
        return self


class TropicalHypersurface(StrictModel):
    """The complete bivariate corner locus and its dual regular subdivision."""

    subdivision: TropicalRegularSubdivision
    cells: tuple[TropicalHypersurfaceCell, ...] = Field(
        min_length=1, max_length=MAX_TROPICAL_HYPERSURFACE_CELLS
    )

    @model_validator(mode="after")
    def require_source_bound_cells(self) -> Self:
        if len(self.subdivision.source.variables) != 2:
            raise _validation_error(
                "hypersurface_source_dimension",
                "a bivariate hypersurface requires a bivariate source polynomial",
            )
        ids = tuple(cell.cell_id for cell in self.cells)
        if len(set(ids)) != len(ids):
            raise _validation_error(
                "hypersurface_cell_ids", "corner-cell identifiers must be unique"
            )
        support_by_id = {
            support.face_id: support for support in self.subdivision.face_supports
        }
        cell_by_id = {cell.cell_id: cell for cell in self.cells}
        expected_dual_faces = {
            support.face_id
            for support in self.subdivision.face_supports
            if support.dimension in (1, 2)
        }
        if {cell.dual_face_id for cell in self.cells} != expected_dual_faces:
            raise _validation_error(
                "hypersurface_face_completeness",
                "every one- or two-dimensional subdivision face needs one dual cell",
            )
        for cell in self.cells:
            support = support_by_id.get(cell.dual_face_id)
            if (
                support is None
                or support.dimension != 2 - cell.dimension
                or support.source_term_indices != cell.active_term_indices
                or support.lifted_face_indices != cell.dual_lifted_face_indices
                or cell.inequalities.space.axes != self.subdivision.source.variables
                or any(
                    index >= len(self.subdivision.source.terms)
                    for index in cell.active_term_indices
                )
                or any(index not in cell_by_id for index in cell.incident_cell_ids)
            ):
                raise _validation_error(
                    "hypersurface_dual_binding",
                    "corner cells must bind to exact source and dual subdivision data",
                )
            for incident_id in cell.incident_cell_ids:
                incident = cell_by_id[incident_id]
                if (
                    abs(incident.dimension - cell.dimension) != 1
                    or cell.cell_id not in incident.incident_cell_ids
                ):
                    raise _validation_error(
                        "hypersurface_incidence",
                        "corner-cell incidence must be reciprocal and codimension one",
                    )
        for edge in self.cells:
            if edge.dimension != 1:
                continue
            for vertex in self.cells:
                if vertex.dimension != 0:
                    continue
                is_incident = set(edge.active_term_indices).issubset(
                    vertex.active_term_indices
                )
                if (vertex.cell_id in edge.incident_cell_ids) != is_incident:
                    raise _validation_error(
                        "hypersurface_incidence_completeness",
                        "edge-to-vertex incidence must match dual face containment",
                    )
        return self


__all__ = [
    "MAX_TROPICAL_ACTIVE_RESULT_BYTES",
    "MAX_TROPICAL_ACTIVE_TERM_WORK",
    "MAX_TROPICAL_ESSENTIAL_FACES",
    "MAX_TROPICAL_ESSENTIAL_RESULT_BYTES",
    "MAX_TROPICAL_ESSENTIAL_TERMS",
    "MAX_TROPICAL_ESSENTIAL_VARIABLES",
    "MAX_TROPICAL_ESSENTIAL_WORK",
    "MAX_TROPICAL_EXPONENT",
    "MAX_TROPICAL_HYPERSURFACE_CELLS",
    "MAX_TROPICAL_HYPERSURFACE_RESULT_BYTES",
    "MAX_TROPICAL_MATRIX_CELLS",
    "MAX_TROPICAL_NEWTON_RESULT_BYTES",
    "MAX_TROPICAL_POLYNOMIAL_TERMS",
    "MAX_TROPICAL_ROOT_CROSSOVER_PAIRS",
    "MAX_TROPICAL_ROOT_DIGITS",
    "MAX_TROPICAL_ROOT_RESULT_BYTES",
    "MAX_TROPICAL_SCALAR_DIGITS",
    "MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS",
    "MAX_TROPICAL_SUBDIVISION_RESULT_BYTES",
    "MAX_TROPICAL_SUBDIVISION_TERMS",
    "MAX_TROPICAL_VECTOR_DIMENSION",
    "TropicalEssentialHullFace",
    "TropicalEssentialLiftedFace",
    "TropicalHypersurface",
    "TropicalHypersurfaceCell",
    "TropicalLiftedSubdivisionFace",
    "TropicalMatrix",
    "TropicalNewtonPolygonEdge",
    "TropicalNewtonPolygonProfile",
    "TropicalNewtonPolygonVertex",
    "TropicalPolynomial",
    "TropicalPolynomialEssentialPart",
    "TropicalPolynomialTerm",
    "TropicalRegularSubdivision",
    "TropicalRootBreakpoint",
    "TropicalRootInterval",
    "TropicalScalar",
    "TropicalSemiring",
    "TropicalSubdivisionFaceSupport",
    "TropicalUnivariateRootProfile",
    "TropicalVector",
    "require_scalar_budget",
]
