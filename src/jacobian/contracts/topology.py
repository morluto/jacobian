"""Bounded contracts for exact finite simplicial topology."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.results import ContractModel

MAX_TOPOLOGY_VERTICES = 64
MAX_TOPOLOGY_FACETS = 128
MAX_TOPOLOGY_DIMENSION = 7
MAX_TOPOLOGY_FACES = 2048
MAX_TOPOLOGY_CHAIN_GROUP = 512
MAX_TOPOLOGY_MATRIX_CELLS = 131_072
MAX_TOPOLOGY_PRIME = 251

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


class SimplicialComplexRequest(ContractModel):
    """A bounded facet presentation validated before artifact materialization."""

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


class FacesInDimension(ContractModel):
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


class FiniteSimplicialComplex(ContractModel):
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
        expected_digest = simplicial_complex_digest(
            vertices=self.vertices,
            maximal_simplices=self.maximal_simplices,
            faces_by_dimension=self.faces_by_dimension,
            dimension=self.dimension,
            f_vector=self.f_vector,
            closure_size=self.closure_size,
        )
        if self.complex_digest != expected_digest:
            raise ValueError("complex_digest does not bind the canonical complex")
        return self


class TopologyExactResult(ContractModel):
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["jacobian.topology"] = "jacobian.topology"
    backend_version: Literal["1"] = "1"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"


class SimplicialComplexMaterializationResult(TopologyExactResult):
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


class ChainComplexRequest(ContractModel):
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


class SimplexBasis(ContractModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    simplices: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )


class SparseMatrixEntry(ContractModel):
    row: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    column: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    value: StrictInt = Field(ge=-1, le=MAX_TOPOLOGY_PRIME - 1)


class SparseBoundaryMatrix(ContractModel):
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


class BoundarySquareLedgerEntry(ContractModel):
    upper_dimension: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_DIMENSION)
    product_rows: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    product_columns: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    nonzero_entries: Literal[0] = 0
    product_is_zero: Literal[True] = True


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
        if self.coefficient_ring is ChainCoefficientRing.INTEGER:
            if self.prime is not None:
                raise ValueError("integer result must not declare a prime")
            allowed_values = {-1, 1}
        else:
            if self.prime is None or not is_bounded_prime(self.prime):
                raise ValueError("prime-field result requires a bounded prime")
            allowed_values = set(range(1, self.prime))
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
        if self.convention is HomologyConvention.REDUCED:
            if self.augmentation is None:
                raise ValueError("reduced chains require the augmentation map")
        elif self.augmentation is not None:
            raise ValueError("unreduced chains must not include an augmentation")
        expected_ledger = tuple(range(1, len(self.simplex_bases)))
        if tuple(item.upper_dimension for item in self.boundary_squared_zero) != (
            expected_ledger
        ):
            raise ValueError("boundary-square ledger must cover every adjacent pair")
        return self


class SimplicialHomologyRequest(ContractModel):
    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_prime_and_bounds(self) -> Self:
        if not is_bounded_prime(self.prime):
            raise ValueError("homology coefficients require a bounded prime")
        require_linear_algebra_bounds(self.complex)
        return self


class ModularVector(ContractModel):
    coefficients: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )


class HomologyGroupResult(ContractModel):
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


__all__ = [
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
    "ModularVector",
    "Simplex",
    "SimplexBasis",
    "SimplicialComplexMaterializationResult",
    "SimplicialComplexRequest",
    "SimplicialHomologyRequest",
    "SimplicialHomologyResult",
    "SparseBoundaryMatrix",
    "SparseMatrixEntry",
    "VertexLabel",
    "canonical_simplex",
    "face_closure",
    "is_bounded_prime",
    "require_linear_algebra_bounds",
    "simplicial_complex_digest",
]
