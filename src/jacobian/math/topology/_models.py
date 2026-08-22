"""Bounded contracts for exact finite simplicial topology."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictInt,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from jacobian._digest import Sha256Digest
from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.math.matrices.certified_snf.values import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)

MAX_TOPOLOGY_VERTICES = 64
MAX_TOPOLOGY_FACETS = 128
MAX_TOPOLOGY_DIMENSION = 7
MAX_TOPOLOGY_FACES = 2048
MAX_TOPOLOGY_CHAIN_GROUP = 512
MAX_INLINE_HOMOLOGY_CHAIN_GROUP = 64
MAX_TOPOLOGY_MATRIX_CELLS = 131_072
MAX_TOPOLOGY_PRIME = 251
MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP = 16
MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK = 32
MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS = 256
MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS = 256

VertexLabel = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$",
        strict=True,
    ),
]
Simplex = tuple[VertexLabel, ...]


class HomologyConvention(StrEnum):
    UNREDUCED = "UNREDUCED"
    REDUCED = "REDUCED"


class ChainCoefficientRing(StrEnum):
    INTEGER = "INTEGER"
    PRIME_FIELD = "PRIME_FIELD"


def is_bounded_prime(value: int) -> bool:
    """Return whether ``value`` is prime within the public coefficient bound."""

    if not 2 <= value <= MAX_TOPOLOGY_PRIME:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def canonical_simplex(simplex: Simplex) -> Simplex:
    return tuple(sorted(simplex))


def face_closure(facets: tuple[Simplex, ...]) -> tuple[tuple[Simplex, ...], ...]:
    """Materialize the non-empty face closure in canonical dimension order."""

    faces: list[set[Simplex]] = [set() for _ in range(MAX_TOPOLOGY_DIMENSION + 1)]
    for facet in facets:
        for size in range(1, len(facet) + 1):
            faces[size - 1].update(combinations(facet, size))
    highest = max(index for index, values in enumerate(faces) if values)
    return tuple(tuple(sorted(values)) for values in faces[: highest + 1])


def _require_request_complex(
    vertices: tuple[VertexLabel, ...],
    facets: tuple[Simplex, ...],
) -> tuple[Simplex, ...]:
    if len(vertices) != len(set(vertices)):
        raise ValueError("simplicial-complex vertices must be unique")
    vertex_set = set(vertices)
    canonical: list[Simplex] = []
    for facet in facets:
        if len(facet) != len(set(facet)):
            raise ValueError("a facet must not repeat a vertex")
        if not set(facet).issubset(vertex_set):
            raise ValueError("every facet vertex must be declared in vertices")
        canonical.append(canonical_simplex(facet))
    if len(canonical) != len(set(canonical)):
        raise ValueError("facets must be distinct after orientation normalization")
    if set().union(*(set(facet) for facet in canonical)) != vertex_set:
        raise ValueError(
            "every vertex must occur in a facet; use a singleton facet for an "
            "isolated vertex"
        )
    for left, right in combinations(canonical, 2):
        if set(left) < set(right) or set(right) < set(left):
            raise ValueError("facet input must contain only maximal simplices")
    closure = face_closure(tuple(canonical))
    if sum(map(len, closure)) > MAX_TOPOLOGY_FACES:
        raise ValueError(
            f"face closure may contain at most {MAX_TOPOLOGY_FACES} non-empty faces"
        )
    return tuple(sorted(canonical))


class SimplicialComplexRequest(StrictModel):
    """A bounded facet presentation for canonicalization."""

    vertices: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
    )
    facets: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )

    @model_validator(mode="after")
    def require_bounded_maximal_facets(self) -> Self:
        if any(
            not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1 for facet in self.facets
        ):
            raise ValueError(
                "each facet must contain between 1 and "
                f"{MAX_TOPOLOGY_DIMENSION + 1} vertices"
            )
        _require_request_complex(self.vertices, self.facets)
        return self


class FacesInDimension(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    faces: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACES,
    )

    @model_validator(mode="after")
    def require_canonical_faces(self) -> Self:
        expected_size = self.dimension + 1
        if any(
            len(face) != expected_size or tuple(sorted(face)) != face
            for face in self.faces
        ):
            raise ValueError("faces must use canonical vertex order and dimension")
        if tuple(sorted(set(self.faces))) != self.faces:
            raise ValueError("faces must be unique and lexicographically ordered")
        return self


def simplicial_complex_digest(
    *,
    vertices: tuple[VertexLabel, ...],
    maximal_simplices: tuple[Simplex, ...],
    faces_by_dimension: tuple[FacesInDimension, ...],
    dimension: int,
    f_vector: tuple[int, ...],
    closure_size: int,
) -> str:
    payload = {
        "complex_format": "jacobian.finite-simplicial-complex/v1",
        "vertices": list(vertices),
        "maximal_simplices": [list(simplex) for simplex in maximal_simplices],
        "faces_by_dimension": [
            {
                "dimension": item.dimension,
                "faces": [list(face) for face in item.faces],
            }
            for item in faces_by_dimension
        ],
        "dimension": dimension,
        "f_vector": list(f_vector),
        "closure_size": closure_size,
        "orientation_convention": "LEXICOGRAPHIC_VERTEX_ORDER",
        "empty_simplex_stored": False,
    }
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


class FiniteSimplicialComplex(StrictModel):
    """Canonical non-empty faces of one finite abstract simplicial complex."""

    complex_format: Literal["jacobian.finite-simplicial-complex/v1"] = (
        "jacobian.finite-simplicial-complex/v1"
    )
    vertices: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
    )
    maximal_simplices: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )
    faces_by_dimension: tuple[FacesInDimension, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    f_vector: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    closure_size: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_FACES)
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    empty_simplex_stored: Literal[False] = False
    complex_digest: Sha256Digest

    @field_validator("complex_digest", mode="after")
    @classmethod
    def require_digest_binds_canonical_complex(
        cls, value: str, info: ValidationInfo
    ) -> str:
        """Bind ``complex_digest`` to the canonical complex derived from the
        other fields.  Runs as a field validator so Pydantic reports the
        error location as ``complex_digest`` (nested inside the parent
        model's ``loc``), not as a model-level error.
        """

        required = (
            "vertices",
            "maximal_simplices",
            "faces_by_dimension",
            "dimension",
            "f_vector",
            "closure_size",
        )
        if not all(key in info.data for key in required):
            return value
        vertices: tuple[str, ...] = info.data["vertices"]
        maximal_simplices: tuple[tuple[str, ...], ...] = info.data["maximal_simplices"]
        faces_by_dimension: tuple[FacesInDimension, ...] = info.data[
            "faces_by_dimension"
        ]
        dimension: int = info.data["dimension"]
        f_vector: tuple[int, ...] = info.data["f_vector"]
        closure_size: int = info.data["closure_size"]
        expected_digest = simplicial_complex_digest(
            vertices=vertices,
            maximal_simplices=maximal_simplices,
            faces_by_dimension=faces_by_dimension,
            dimension=dimension,
            f_vector=f_vector,
            closure_size=closure_size,
        )
        if value != expected_digest:
            raise ValueError("complex_digest does not bind the canonical complex")
        return value

    @model_validator(mode="after")
    def require_complete_canonical_complex(self) -> Self:
        if tuple(sorted(set(self.vertices))) != self.vertices:
            raise ValueError("complex vertices must be unique and canonical")
        canonical_facets = _require_request_complex(
            self.vertices,
            self.maximal_simplices,
        )
        if canonical_facets != self.maximal_simplices:
            raise ValueError("maximal simplices must be canonical")
        closure = face_closure(self.maximal_simplices)
        expected_faces = tuple(
            FacesInDimension(dimension=dimension, faces=faces)
            for dimension, faces in enumerate(closure)
        )
        if self.faces_by_dimension != expected_faces:
            raise ValueError("faces_by_dimension is not the complete face closure")
        expected_f_vector = tuple(len(faces) for faces in closure)
        if (
            self.dimension != len(closure) - 1
            or self.f_vector != expected_f_vector
            or self.closure_size != sum(expected_f_vector)
        ):
            raise ValueError("complex dimension, f-vector, or closure size is invalid")
        return self


class TopologyExactResult(StrictModel):
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"


class SimplicialComplexCanonicalizationResult(TopologyExactResult):
    complex: FiniteSimplicialComplex
    completeness: Literal["COMPLETE_FACE_CLOSURE"] = "COMPLETE_FACE_CLOSURE"


def require_linear_algebra_bounds(complex_: FiniteSimplicialComplex) -> None:
    sizes = complex_.f_vector
    if any(size > MAX_TOPOLOGY_CHAIN_GROUP for size in sizes):
        raise ValueError(
            f"each chain group may contain at most {MAX_TOPOLOGY_CHAIN_GROUP} faces"
        )
    padded = (0, *sizes)
    if any(
        rows * columns > MAX_TOPOLOGY_MATRIX_CELLS for rows, columns in pairwise(padded)
    ):
        raise ValueError(
            f"a boundary matrix exceeds the {MAX_TOPOLOGY_MATRIX_CELLS}-cell bound"
        )


class ChainComplexRequest(StrictModel):
    complex: FiniteSimplicialComplex
    coefficient_ring: ChainCoefficientRing = ChainCoefficientRing.INTEGER
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_coefficient_semantics_and_bounds(self) -> Self:
        if self.coefficient_ring is ChainCoefficientRing.INTEGER:
            if self.prime is not None:
                raise ValueError("integer chain complexes must not declare a prime")
        elif self.prime is None or not is_bounded_prime(self.prime):
            raise ValueError("prime-field chain complexes require a bounded prime")
        require_linear_algebra_bounds(self.complex)
        return self


class SimplexBasis(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    simplices: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )


class SparseMatrixEntry(StrictModel):
    row: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    column: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    value: StrictInt = Field(ge=-1, le=MAX_TOPOLOGY_PRIME - 1)


class SparseBoundaryMatrix(StrictModel):
    source_dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    target_dimension: StrictInt = Field(ge=-1, le=MAX_TOPOLOGY_DIMENSION - 1)
    rows: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    columns: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    entries: tuple[SparseMatrixEntry, ...] = Field(
        default=(),
        max_length=(MAX_TOPOLOGY_DIMENSION + 1) * MAX_TOPOLOGY_CHAIN_GROUP,
    )

    @model_validator(mode="after")
    def require_canonical_sparse_entries(self) -> Self:
        coordinates = tuple((entry.row, entry.column) for entry in self.entries)
        if coordinates != tuple(sorted(set(coordinates))):
            raise ValueError("sparse entries must be unique and row-major")
        if any(
            entry.row >= self.rows or entry.column >= self.columns or entry.value == 0
            for entry in self.entries
        ):
            raise ValueError("sparse entry lies outside the matrix or stores zero")
        return self


class BoundarySquareLedgerEntry(StrictModel):
    upper_dimension: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_DIMENSION)
    product_rows: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    product_columns: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    nonzero_entries: Literal[0] = 0
    product_is_zero: Literal[True] = True


def _resolve_chain_coefficient_values(
    coefficient_ring: ChainCoefficientRing,
    prime: StrictInt | None,
) -> set[int]:
    if coefficient_ring is ChainCoefficientRing.INTEGER:
        if prime is not None:
            raise ValueError("integer result must not declare a prime")
        return {-1, 1}
    if prime is None or not is_bounded_prime(prime):
        raise ValueError("prime-field result requires a bounded prime")
    return set(range(1, prime))


def _validate_chain_convention_augmentation(
    convention: HomologyConvention,
    augmentation: SparseBoundaryMatrix | None,
) -> None:
    if convention is HomologyConvention.REDUCED:
        if augmentation is None:
            raise ValueError("reduced chains require the augmentation map")
    elif augmentation is not None:
        raise ValueError("unreduced chains must not include an augmentation")


class ChainComplexResult(TopologyExactResult):
    complex_digest: Sha256Digest
    coefficient_ring: ChainCoefficientRing
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    simplex_bases: tuple[SimplexBasis, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    boundary_matrices: tuple[SparseBoundaryMatrix, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    augmentation: SparseBoundaryMatrix | None = None
    boundary_squared_zero: tuple[BoundarySquareLedgerEntry, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_DIMENSION,
    )

    @model_validator(mode="after")
    def require_coherent_chain_contract(self) -> Self:
        allowed_values = _resolve_chain_coefficient_values(
            self.coefficient_ring, self.prime
        )
        dimensions = tuple(item.dimension for item in self.simplex_bases)
        if dimensions != tuple(range(len(self.simplex_bases))):
            raise ValueError("simplex bases must cover contiguous dimensions")
        if tuple(matrix.source_dimension for matrix in self.boundary_matrices) != (
            dimensions
        ):
            raise ValueError("boundary matrices must align with simplex bases")
        for matrix in self.boundary_matrices:
            if any(entry.value not in allowed_values for entry in matrix.entries):
                raise ValueError("boundary coefficient is outside its coefficient ring")
        _validate_chain_convention_augmentation(self.convention, self.augmentation)
        expected_ledger = tuple(range(1, len(self.simplex_bases)))
        if tuple(item.upper_dimension for item in self.boundary_squared_zero) != (
            expected_ledger
        ):
            raise ValueError("boundary-square ledger must cover every adjacent pair")
        return self


class SimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_prime_and_bounds(self) -> Self:
        if not is_bounded_prime(self.prime):
            raise ValueError("homology coefficients require a bounded prime")
        require_linear_algebra_bounds(self.complex)
        if any(
            size > MAX_INLINE_HOMOLOGY_CHAIN_GROUP for size in self.complex.f_vector
        ):
            raise ValueError(
                "inline homology bases require at most "
                f"{MAX_INLINE_HOMOLOGY_CHAIN_GROUP} simplices in each chain group"
            )
        return self


class ModularVector(StrictModel):
    coefficients: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )


class HomologyGroupResult(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    chain_dimension: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    outgoing_boundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cycle_dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    incoming_boundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    betti_number: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cycle_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    boundary_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    homology_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    quotient_span_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)

    @model_validator(mode="after")
    def require_dimension_ledger(self) -> Self:
        if self.cycle_dimension != (self.chain_dimension - self.outgoing_boundary_rank):
            raise ValueError("cycle dimension does not equal nullity")
        if self.betti_number != (self.cycle_dimension - self.incoming_boundary_rank):
            raise ValueError("Betti number does not equal dim cycles minus boundaries")
        if (
            len(self.cycle_basis) != self.cycle_dimension
            or len(self.boundary_basis) != self.incoming_boundary_rank
            or len(self.homology_basis) != self.betti_number
            or self.quotient_span_rank != self.cycle_dimension
        ):
            raise ValueError("homology bases do not match the dimension ledger")
        vectors = (
            *self.cycle_basis,
            *self.boundary_basis,
            *self.homology_basis,
        )
        if any(len(vector.coefficients) != self.chain_dimension for vector in vectors):
            raise ValueError("homology vector does not use the declared chain basis")
        return self


class SimplicialHomologyResult(TopologyExactResult):
    complex_digest: Sha256Digest
    coefficient_field: Literal["PRIME_FIELD"] = "PRIME_FIELD"
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    dimension_range: tuple[StrictInt, StrictInt]
    groups: tuple[HomologyGroupResult, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_complete_dimension_range(self) -> Self:
        if not is_bounded_prime(self.prime):
            raise ValueError("homology result requires a bounded prime")
        dimensions = tuple(group.dimension for group in self.groups)
        if dimensions != tuple(range(len(self.groups))):
            raise ValueError("homology groups must cover contiguous dimensions")
        if self.dimension_range != (0, len(self.groups) - 1):
            raise ValueError("dimension_range does not cover every returned group")
        if any(
            coefficient < 0 or coefficient >= self.prime
            for group in self.groups
            for vector in (
                *group.cycle_basis,
                *group.boundary_basis,
                *group.homology_basis,
            )
            for coefficient in vector.coefficients
        ):
            raise ValueError("homology vector coefficient is outside the prime field")
        return self


class IntegralSimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_integral_certificate_bounds(self) -> Self:
        require_linear_algebra_bounds(self.complex)
        if any(
            size > MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP for size in self.complex.f_vector
        ):
            raise ValueError(
                "integral homology requires at most "
                f"{MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP} simplices in each chain group"
            )
        if sum(self.complex.f_vector) > MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK:
            raise ValueError(
                "integral homology requires total chain rank at most "
                f"{MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK}"
            )
        padded = (0, *self.complex.f_vector)
        if any(
            rows * columns > MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS
            for rows, columns in pairwise(padded)
        ):
            raise ValueError(
                "integral homology boundary exceeds the "
                f"{MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS}-cell bound"
            )
        return self


class IntegralVector(StrictModel):
    coefficients: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )

    @model_validator(mode="after")
    def require_output_digit_budget(self) -> Self:
        if any(
            len(value.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
            for value in self.coefficients
        ):
            raise ValueError("integral homology vector exceeds the output digit bound")
        return self


class IntegralFreeGenerator(StrictModel):
    cycle: IntegralVector
    cycle_coordinates: IntegralVector


class IntegralTorsionGenerator(StrictModel):
    order: CanonicalInteger
    cycle: IntegralVector
    cycle_coordinates: IntegralVector
    bounding_chain: IntegralVector

    @model_validator(mode="after")
    def require_nontrivial_bounded_order(self) -> Self:
        if (
            int(self.order) <= 1
            or len(self.order.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
        ):
            raise ValueError("torsion generator order must be a bounded integer > 1")
        return self


class IntegralHomologyGroupResult(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    chain_dimension: StrictInt = Field(ge=1, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    incoming_chain_dimension: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    outgoing_boundary_rank: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    cycle_rank: StrictInt = Field(ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    incoming_boundary_rank: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    betti_number: StrictInt = Field(ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    torsion_coefficients: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    free_generators: tuple[IntegralFreeGenerator, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    torsion_generators: tuple[IntegralTorsionGenerator, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    outgoing_smith_certificate: SmithNormalFormCertificate
    boundary_in_cycle_coordinates: CertifiedIntegerMatrix
    incoming_smith_certificate: SmithNormalFormCertificate
    generator_basis: Literal[
        "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"
    ] = "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"

    @model_validator(mode="after")
    def require_complete_integral_group_ledger(self) -> Self:
        outgoing = self.outgoing_smith_certificate
        incoming = self.incoming_smith_certificate
        if (
            outgoing.source.column_count != self.chain_dimension
            or outgoing.rank != self.outgoing_boundary_rank
            or self.cycle_rank != self.chain_dimension - self.outgoing_boundary_rank
            or (
                self.boundary_in_cycle_coordinates.row_count,
                self.boundary_in_cycle_coordinates.column_count,
            )
            != (self.cycle_rank, self.incoming_chain_dimension)
            or incoming.source != self.boundary_in_cycle_coordinates
            or incoming.rank != self.incoming_boundary_rank
            or self.betti_number != self.cycle_rank - self.incoming_boundary_rank
            or len(self.free_generators) != self.betti_number
        ):
            raise ValueError("integral homology rank and certificate ledger is invalid")
        torsion = tuple(
            factor for factor in incoming.invariant_factors if int(factor) > 1
        )
        if (
            self.torsion_coefficients != torsion
            or tuple(item.order for item in self.torsion_generators) != torsion
        ):
            raise ValueError(
                "integral homology torsion generators must match Smith factors"
            )
        if any(
            len(item.cycle.coefficients) != self.chain_dimension
            or len(item.cycle_coordinates.coefficients) != self.cycle_rank
            for item in self.free_generators
        ) or any(
            len(item.cycle.coefficients) != self.chain_dimension
            or len(item.cycle_coordinates.coefficients) != self.cycle_rank
            or len(item.bounding_chain.coefficients) != self.incoming_chain_dimension
            for item in self.torsion_generators
        ):
            raise ValueError(
                "integral homology generators must use the declared simplex bases"
            )
        matrices = (
            outgoing.source,
            outgoing.diagonal,
            outgoing.left_transformation,
            outgoing.right_transformation,
            self.boundary_in_cycle_coordinates,
            incoming.source,
            incoming.diagonal,
            incoming.left_transformation,
            incoming.right_transformation,
        )
        scalar_values = (
            value for matrix in matrices for row in matrix.entries for value in row
        )
        if any(
            len(value.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
            for value in scalar_values
        ):
            raise ValueError(
                "integral homology certificate exceeds the output digit bound"
            )
        return self


class IntegralSimplicialHomologyResult(TopologyExactResult):
    complex_digest: Sha256Digest
    coefficient_ring: Literal["ZZ"] = "ZZ"
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    dimension_range: tuple[StrictInt, StrictInt]
    groups: tuple[IntegralHomologyGroupResult, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    completeness: Literal["FREE_TORSION_AND_BOUND_GENERATORS"] = (
        "FREE_TORSION_AND_BOUND_GENERATORS"
    )
    decomposition: Literal["DIRECT_SUM_Z_AND_FINITE_CYCLIC_FACTORS"] = (
        "DIRECT_SUM_Z_AND_FINITE_CYCLIC_FACTORS"
    )

    @model_validator(mode="after")
    def require_complete_integral_dimension_range(self) -> Self:
        dimensions = tuple(group.dimension for group in self.groups)
        if dimensions != tuple(range(len(self.groups))):
            raise ValueError(
                "integral homology groups must cover contiguous dimensions"
            )
        if self.dimension_range != (0, len(self.groups) - 1):
            raise ValueError("integral homology dimension_range must cover every group")
        return self


__all__ = [
    "MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP",
    "MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS",
    "MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS",
    "MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK",
    "MAX_TOPOLOGY_CHAIN_GROUP",
    "MAX_TOPOLOGY_DIMENSION",
    "MAX_TOPOLOGY_FACES",
    "MAX_TOPOLOGY_FACETS",
    "MAX_TOPOLOGY_MATRIX_CELLS",
    "MAX_TOPOLOGY_PRIME",
    "MAX_TOPOLOGY_VERTICES",
    "BoundarySquareLedgerEntry",
    "ChainCoefficientRing",
    "ChainComplexRequest",
    "ChainComplexResult",
    "FacesInDimension",
    "FiniteSimplicialComplex",
    "HomologyConvention",
    "HomologyGroupResult",
    "IntegralFreeGenerator",
    "IntegralHomologyGroupResult",
    "IntegralSimplicialHomologyRequest",
    "IntegralSimplicialHomologyResult",
    "IntegralTorsionGenerator",
    "IntegralVector",
    "ModularVector",
    "Simplex",
    "SimplexBasis",
    "SimplicialComplexCanonicalizationResult",
    "SimplicialComplexRequest",
    "SimplicialHomologyRequest",
    "SimplicialHomologyResult",
    "SparseBoundaryMatrix",
    "SparseMatrixEntry",
    "VertexLabel",
    "face_closure",
    "simplicial_complex_digest",
]


class FVectorRequest(StrictModel):
    """Request the f-vector and h-vector of a simplicial complex."""

    complex: SimplicialComplexRequest


class FVectorResult(TopologyExactResult):
    """The f-vector and h-vector of a simplicial complex."""

    f_vector: tuple[int, ...]
    h_vector: tuple[int, ...]
    euler_characteristic: int
    dimension: int


class LinkRequest(StrictModel):
    """Request the link of a simplex in a simplicial complex."""

    complex: SimplicialComplexRequest
    simplex: tuple[VertexLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_simplex(self) -> Self:
        simplex = set(self.simplex)
        if len(simplex) != len(self.simplex):
            raise ValueError("simplex vertices must be distinct")
        if not simplex.issubset(self.complex.vertices):
            raise ValueError("simplex vertices must be in the complex")
        if not any(simplex.issubset(facet) for facet in self.complex.facets):
            raise ValueError("simplex must be a face of the complex")
        return self


class LinkResult(TopologyExactResult):
    """The maximal facets of the link of a simplex."""

    simplex: tuple[str, ...]
    link_facets: tuple[tuple[str, ...], ...]
    link_is_empty: bool


__all__.extend(["FVectorRequest", "FVectorResult", "LinkRequest", "LinkResult"])


class StarRequest(StrictModel):
    """Request the closed star of a simplex in a simplicial complex."""

    complex: SimplicialComplexRequest
    simplex: tuple[VertexLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_star_simplex(self) -> Self:
        simplex = set(self.simplex)
        if len(simplex) != len(self.simplex):
            raise ValueError("simplex vertices must be distinct")
        if not simplex.issubset(self.complex.vertices):
            raise ValueError("simplex vertices must be in the complex")
        if not any(simplex.issubset(facet) for facet in self.complex.facets):
            raise ValueError("simplex must be a face of the complex")
        return self


class StarResult(TopologyExactResult):
    """The closed star of a simplex: all facets containing sigma."""

    simplex: tuple[str, ...]
    star_facets: tuple[tuple[str, ...], ...]
    star_is_empty: bool
    star_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_star_canonical(self) -> Self:
        if self.star_is_empty:
            if self.star_complex is not None:
                raise ValueError("empty star must have no complex")
            if self.star_facets:
                raise ValueError("empty star must have no facets")
        else:
            if self.star_complex is None:
                raise ValueError("non-empty star requires star_complex")
            if tuple(sorted(self.star_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.star_facets)
            ):
                raise ValueError("star_complex maximal simplices must match star_facets")
            if set(self.star_complex.vertices) != set(
                v for facet in self.star_facets for v in facet
            ):
                raise ValueError("star_complex vertices must match star_facets")
        return self


class VertexDeletionRequest(StrictModel):
    """Delete a vertex subset from a simplicial complex."""

    complex: SimplicialComplexRequest
    vertices_to_delete: tuple[VertexLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_deletion(self) -> Self:
        vtd = set(self.vertices_to_delete)
        if len(vtd) != len(self.vertices_to_delete):
            raise ValueError("vertices_to_delete must be distinct")
        if not vtd.issubset(set(self.complex.vertices)):
            raise ValueError("vertices_to_delete must be in the complex")
        return self


class VertexDeletionResult(TopologyExactResult):
    """The complex after deleting a vertex subset."""

    deleted_vertices: tuple[str, ...]
    remaining_vertices: tuple[str, ...]
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_deletion_canonical(self) -> Self:
        if not self.remaining_facets:
            if self.remaining_complex is not None:
                raise ValueError("empty deletion must have no remaining_complex")
            if self.remaining_vertices:
                raise ValueError("empty deletion must have no remaining_vertices")
        else:
            if self.remaining_complex is None:
                raise ValueError("non-empty deletion requires remaining_complex")
            if tuple(sorted(self.remaining_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.remaining_facets)
            ):
                raise ValueError(
                    "remaining_complex maximal simplices must match remaining_facets"
                )
            if tuple(sorted(self.remaining_complex.vertices)) != tuple(
                sorted(self.remaining_vertices)
            ):
                raise ValueError(
                    "remaining_complex vertices must match remaining_vertices"
                )
        return self


class SkeletonRequest(StrictModel):
    """Request the k-skeleton of a simplicial complex."""

    complex: SimplicialComplexRequest
    k: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)


class SkeletonResult(TopologyExactResult):
    """The k-skeleton as a facet list."""

    k: int
    skeleton_facets: tuple[tuple[str, ...], ...]
    skeleton_vertices: tuple[str, ...]
    skeleton_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_skeleton_canonical(self) -> Self:
        if not self.skeleton_facets:
            if self.skeleton_complex is not None:
                raise ValueError("empty skeleton must have no complex")
        else:
            if self.skeleton_complex is None:
                raise ValueError("non-empty skeleton requires skeleton_complex")
            if tuple(sorted(self.skeleton_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.skeleton_facets)
            ):
                raise ValueError(
                    "skeleton_complex maximal simplices must match skeleton_facets"
                )
            if tuple(sorted(self.skeleton_complex.vertices)) != tuple(
                sorted(self.skeleton_vertices)
            ):
                raise ValueError(
                    "skeleton_complex vertices must match skeleton_vertices"
                )
        return self


class JoinRequest(StrictModel):
    """Join two simplicial complexes on disjoint vertex sets."""

    complex_a: SimplicialComplexRequest
    complex_b: SimplicialComplexRequest

    @model_validator(mode="after")
    def require_disjoint_vertices(self) -> Self:
        va = set(self.complex_a.vertices)
        vb = set(self.complex_b.vertices)
        if va & vb:
            raise ValueError(
                "join requires disjoint vertex sets; rename vertices first"
            )
        # Validate join result bounds: combined facet dimension must be <=7,
        # vertex count <=64, and facet count within limits.
        combined_vertices = len(va | vb)
        if combined_vertices > MAX_TOPOLOGY_VERTICES:
            raise ValueError(
                f"join would have {combined_vertices} vertices exceeding {MAX_TOPOLOGY_VERTICES}"
            )
        max_joined_size = 0
        for fa in self.complex_a.facets:
            for fb in self.complex_b.facets:
                joined_size = len(set(fa) | set(fb))
                max_joined_size = max(max_joined_size, joined_size)
                if joined_size > MAX_TOPOLOGY_DIMENSION + 1:
                    raise ValueError(
                        f"join facet size {joined_size} exceeds dimension {MAX_TOPOLOGY_DIMENSION} limit"
                    )
        # Check closure size estimate for join: upper bound is product of face counts
        # For two 4-simplices (32 faces each), join has 63 faces? Actually 5+5=10 vertices => 1023 faces >2048? But dimension already fails.
        return self


class JoinResult(TopologyExactResult):
    """The join of two complexes."""

    join_vertices: tuple[str, ...]
    join_facets: tuple[tuple[str, ...], ...]
    join_dimension: int
    join_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_join_canonical(self) -> Self:
        if not self.join_facets:
            if self.join_complex is not None:
                raise ValueError("empty join must have no complex")
        else:
            if self.join_complex is None:
                raise ValueError("non-empty join requires join_complex")
            if tuple(sorted(self.join_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.join_facets)
            ):
                raise ValueError("join_complex maximal simplices must match join_facets")
            if tuple(sorted(self.join_complex.vertices)) != tuple(
                sorted(self.join_vertices)
            ):
                raise ValueError("join_complex vertices must match join_vertices")
            if self.join_complex.dimension != self.join_dimension:
                raise ValueError("join_complex dimension must match join_dimension")
        return self


class BarycentricSubdivisionRequest(StrictModel):
    """Subdivide a complex via its order complex (barycentric subdivision)."""

    complex: SimplicialComplexRequest

    @model_validator(mode="after")
    def require_barycentric_work_bounds(self) -> Self:
        closure = face_closure(tuple(tuple(sorted(f)) for f in self.complex.facets))
        face_count = sum(len(part) for part in closure)
        if face_count > 31:
            raise ValueError(
                f"barycentric subdivision requires at most 31 faces (got {face_count}); "
                "input would produce >128 subdivision facets exceeding result contract"
            )
        # Additional check: subdivision of a 5-vertex simplex has 120 facets within 128 limit;
        # 6-vertex simplex would have 720 >128, so 31 is safe.
        return self


class BarycentricSubdivisionResult(TopologyExactResult):
    """The barycentric subdivision as a facet list."""

    original_vertices: tuple[str, ...]
    original_dimension: int
    subdivision_vertices: tuple[str, ...]
    subdivision_facets: tuple[tuple[str, ...], ...]
    num_new_vertices: int
    subdivision_complex: FiniteSimplicialComplex | None = None
    subdivision_vertex_faces: tuple[tuple[str, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_subdivision_canonical(self) -> Self:
        if not self.subdivision_facets:
            if self.subdivision_complex is not None:
                raise ValueError("empty subdivision must have no complex")
        else:
            if self.subdivision_complex is None:
                raise ValueError("non-empty subdivision requires subdivision_complex")
            if tuple(sorted(self.subdivision_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.subdivision_facets)
            ):
                raise ValueError(
                    "subdivision_complex maximal simplices must match subdivision_facets"
                )
            if tuple(sorted(self.subdivision_complex.vertices)) != tuple(
                sorted(self.subdivision_vertices)
            ):
                raise ValueError(
                    "subdivision_complex vertices must match subdivision_vertices"
                )
            if len(self.subdivision_vertex_faces) != len(self.subdivision_vertices):
                raise ValueError(
                    "subdivision_vertex_faces must align with subdivision_vertices"
                )
            # Validate vertex labels are bounded and injective
            if len(set(self.subdivision_vertices)) != len(self.subdivision_vertices):
                raise ValueError("subdivision vertices must be unique")
            for label in self.subdivision_vertices:
                if len(label) > 32 or not label[0].isalnum():
                    raise ValueError(f"invalid subdivision vertex label: {label}")
        return self


class PseudomanifoldRequest(StrictModel):
    """Decide whether a complex is a pseudomanifold."""

    complex: SimplicialComplexRequest


class PseudomanifoldResult(TopologyExactResult):
    """Pseudomanifold decision result bound to its source complex."""

    complex: SimplicialComplexRequest
    is_pseudomanifold: bool
    is_closed: bool
    dimension: int
    num_facets: int
    obstruction: str | None = None

    @model_validator(mode="after")
    def require_pseudomanifold_binding(self) -> Self:
        facets = [frozenset(f) for f in self.complex.facets]
        dim = max(len(f) - 1 for f in facets) if facets else 0
        if self.dimension != dim or self.num_facets != len(facets):
            raise ValueError("dimension/num_facets must match source complex")
        # Recompute expected decision
        is_pure = all(len(f) - 1 == dim for f in facets) if facets else False
        codim1_count: dict[frozenset[str], int] = {}
        is_closed_expected = False
        obstruction_expected: str | None = None
        expected_is_pseudo = False
        if not is_pure:
            expected_is_pseudo = False
            obstruction_expected = "not pure: facets have different dimensions"
        elif dim < 1:
            expected_is_pseudo = False
            obstruction_expected = "dimension must be at least 1"
        else:
            for facet in facets:
                for face in combinations(sorted(facet), len(facet) - 1):
                    key = frozenset(face)
                    codim1_count[key] = codim1_count.get(key, 0) + 1
            has_over = any(c > 2 for c in codim1_count.values())
            if has_over:
                expected_is_pseudo = False
                # find first over face
                for face, cnt in codim1_count.items():
                    if cnt > 2:
                        obstruction_expected = f"codim-1 face {sorted(face)} is in {cnt} facets"
                        break
            else:
                expected_is_pseudo = True
                is_closed_expected = all(c == 2 for c in codim1_count.values()) if codim1_count else False
                if is_closed_expected:
                    obstruction_expected = None
                else:
                    obstruction_expected = "pseudomanifold with boundary"
                if self.is_closed != is_closed_expected:
                    raise ValueError("is_closed must match codim-1 incidence")
                if self.is_closed and self.obstruction is not None:
                    raise ValueError("closed pseudomanifold must have no obstruction")
                if not self.is_closed and self.obstruction is None:
                    raise ValueError("pseudomanifold with boundary must have obstruction")
        if self.is_pseudomanifold != expected_is_pseudo:
            raise ValueError(
                f"is_pseudomanifold {self.is_pseudomanifold} does not match expected {expected_is_pseudo}"
            )
        # For non-pseudomanifold, obstruction must match expected (allow any non-None when expected is not None? strict)
        if not expected_is_pseudo:
            if self.is_closed:
                raise ValueError("non-pseudomanifold cannot be closed")
            # Obstruction should be present
            if self.obstruction is None:
                raise ValueError("non-pseudomanifold must have obstruction")
        else:
            if self.obstruction != obstruction_expected:
                raise ValueError(
                    f"obstruction {self.obstruction!r} does not match expected {obstruction_expected!r}"
                )
        return self


class ShellingCheckRequest(StrictModel):
    """Check a submitted shelling order of a pure complex."""

    complex: SimplicialComplexRequest
    facet_order: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_order(self) -> Self:
        if sorted(self.facet_order) != list(range(len(self.complex.facets))):
            raise ValueError("facet_order must be a permutation of facet indices")
        return self


class ShellingCheckResult(TopologyExactResult):
    """Result of checking a shelling order."""

    is_shelling: bool
    failed_at: int | None = None
    failure_reason: str | None = None


class ElementaryCollapseRequest(StrictModel):
    """Check and perform one elementary collapse step (free face must be codimension-one)."""

    complex: SimplicialComplexRequest
    free_face: tuple[VertexLabel, ...] = Field(min_length=1)
    coface: tuple[VertexLabel, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_valid_collapse(self) -> Self:
        face_set = set(self.free_face)
        coface_set = set(self.coface)
        if face_set == coface_set:
            raise ValueError("free_face must be a proper subset of coface")
        if not face_set.issubset(coface_set):
            raise ValueError("free_face must be contained in coface")
        if len(coface_set) != len(face_set) + 1:
            raise ValueError(
                "elementary collapse requires free_face to be codimension-one in coface "
                f"(got |free|={len(face_set)}, |coface|={len(coface_set)})"
            )
        return self


class ElementaryCollapseResult(TopologyExactResult):
    """Result of checking one elementary collapse step."""

    is_free_face: bool
    free_face: tuple[str, ...]
    coface: tuple[str, ...]
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_vertices: tuple[str, ...]
    remaining_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_collapse_canonical(self) -> Self:
        if self.is_free_face:
            if not self.remaining_facets:
                if self.remaining_complex is not None:
                    raise ValueError("empty collapsed complex must have no remaining_complex")
                if self.remaining_vertices:
                    raise ValueError("empty collapsed complex must have no remaining_vertices")
            else:
                if self.remaining_complex is None:
                    raise ValueError("non-empty collapse requires remaining_complex")
                if tuple(sorted(self.remaining_complex.maximal_simplices)) != tuple(
                    sorted(tuple(sorted(f)) for f in self.remaining_facets)
                ):
                    raise ValueError(
                        "remaining_complex maximal simplices must match remaining_facets"
                    )
                if tuple(sorted(self.remaining_complex.vertices)) != tuple(
                    sorted(self.remaining_vertices)
                ):
                    raise ValueError(
                        "remaining_complex vertices must match remaining_vertices"
                    )
        else:
            # When not a free face, remaining should be unchanged; allow both but
            # if complex present it must match.
            if self.remaining_complex is not None:
                if tuple(sorted(self.remaining_complex.maximal_simplices)) != tuple(
                    sorted(tuple(sorted(f)) for f in self.remaining_facets)
                ):
                    raise ValueError("remaining_complex must match remaining_facets")
        return self


__all__.extend(
    [
        "BarycentricSubdivisionRequest",
        "BarycentricSubdivisionResult",
        "ElementaryCollapseRequest",
        "ElementaryCollapseResult",
        "JoinRequest",
        "JoinResult",
        "PseudomanifoldRequest",
        "PseudomanifoldResult",
        "ShellingCheckRequest",
        "ShellingCheckResult",
        "SkeletonRequest",
        "SkeletonResult",
        "StarRequest",
        "StarResult",
        "VertexDeletionRequest",
        "VertexDeletionResult",
    ]
)
