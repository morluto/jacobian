"""Bounded contracts for exact finite simplicial topology."""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

MAX_TOPOLOGY_VERTICES = 64
MAX_TOPOLOGY_FACETS = 128
MAX_TOPOLOGY_DIMENSION = 7
MAX_TOPOLOGY_FACES = 2048
MAX_TOPOLOGY_CHAIN_GROUP = 512
MAX_TOPOLOGY_MATRIX_CELLS = 131_072
MAX_TOPOLOGY_PRIME = 251
# Barycentric subdivision admits only the source face counts whose maximal
# chains fit the canonical facet result bound.
MAX_BARYCENTRIC_SOURCE_FACES = 31


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Return a stable owner-local error for a model invariant.

    The human-readable message remains part of the diagnostic, while the
    machine-readable type is intentionally stable for callers and tests.
    """

    return PydanticCustomError(reason, message)


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

    if any(not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1 for facet in facets):
        raise ValueError(
            f"each facet must contain between 1 and {MAX_TOPOLOGY_DIMENSION + 1} vertices"
        )
    faces: list[set[Simplex]] = [set() for _ in range(MAX_TOPOLOGY_DIMENSION + 1)]
    for facet in facets:
        for size in range(1, len(facet) + 1):
            faces[size - 1].update(combinations(facet, size))
    highest = max(index for index, values in enumerate(faces) if values)
    return tuple(tuple(sorted(values)) for values in faces[: highest + 1])


def canonical_complex(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
    *,
    closure: tuple[tuple[Simplex, ...], ...] | None = None,
) -> FiniteSimplicialComplex:
    """Construct the neutral canonical value for validated facet data.

    Admitted producers may pass the closure they already materialized so
    packaging the canonical value does not repeat the same bounded work.
    """

    canonical_vertices = tuple(sorted(vertices))
    canonical_facets = tuple(sorted(tuple(sorted(facet)) for facet in facets))
    if closure is None:
        closure = face_closure(canonical_facets)
    faces_by_dimension = tuple(
        FacesInDimension(dimension=dimension, faces=faces)
        for dimension, faces in enumerate(closure)
    )
    f_vector = tuple(len(faces) for faces in closure)
    closure_size = sum(f_vector)
    dimension = len(closure) - 1
    return FiniteSimplicialComplex(
        vertices=canonical_vertices,
        maximal_simplices=canonical_facets,
        faces_by_dimension=faces_by_dimension,
        dimension=dimension,
        f_vector=f_vector,
        closure_size=closure_size,
    )


def _all_faces(facets: tuple[Simplex, ...]) -> set[tuple[str, ...]]:
    """Return the complete set of nonempty faces for a facet list."""
    faces: set[tuple[str, ...]] = set()
    for facet in facets:
        n = len(facet)
        for r in range(1, n + 1):
            for subset in combinations(facet, r):
                faces.add(tuple(sorted(subset)))
    return faces


def _require_request_complex(
    vertices: tuple[VertexLabel, ...],
    facets: tuple[Simplex, ...],
    *,
    check_closure: bool = True,
) -> tuple[Simplex, ...]:
    if len(vertices) != len(set(vertices)):
        raise _validation_error(
            "topology.require_request_complex_1",
            "simplicial-complex vertices must be unique",
        )
    vertex_set = set(vertices)
    canonical: list[Simplex] = []
    for facet in facets:
        if len(facet) != len(set(facet)):
            raise _validation_error(
                "topology.require_request_complex_2", "a facet must not repeat a vertex"
            )
        if not set(facet).issubset(vertex_set):
            raise _validation_error(
                "topology.require_request_complex_3",
                "every facet vertex must be declared in vertices",
            )
        canonical.append(canonical_simplex(facet))
    if len(canonical) != len(set(canonical)):
        raise _validation_error(
            "topology.require_request_complex_4",
            "facets must be distinct after orientation normalization",
        )
    if set().union(*(set(facet) for facet in canonical)) != vertex_set:
        raise _validation_error(
            "topology.require_request_complex_5",
            "every vertex must occur in a facet; use a singleton facet for an "
            "isolated vertex",
        )
    for left, right in combinations(canonical, 2):
        if set(left) < set(right) or set(right) < set(left):
            raise _validation_error(
                "topology.require_request_complex_6",
                "facet input must contain only maximal simplices",
            )
    if check_closure:
        closure = face_closure(tuple(canonical))
        if sum(map(len, closure)) > MAX_TOPOLOGY_FACES:
            raise _validation_error(
                "topology.require_request_complex_7",
                f"face closure may contain at most {MAX_TOPOLOGY_FACES} non-empty faces",
            )
    return tuple(sorted(canonical))


class SimplicialComplexRequest(StrictModel):
    """A bounded facet presentation or an unchanged canonical complex.

    Canonical input is structurally decoded and projected to its maximal facets.
    Operations establish the face closure from those facets during admission.
    """

    vertices: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
    )
    facets: tuple[Simplex, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )

    @model_validator(mode="before")
    @classmethod
    def accept_canonical_complex_value(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        if isinstance(data, dict) and "maximal_simplices" in data:
            data = FiniteSimplicialComplex.model_validate(data)
        if isinstance(data, FiniteSimplicialComplex):
            return {"vertices": data.vertices, "facets": data.maximal_simplices}
        return data

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> JsonSchemaValue:
        facet_presentation = handler(core_schema)
        canonical_value = handler(FiniteSimplicialComplex.__pydantic_core_schema__)
        facet_presentation.pop("title", None)
        return {
            "type": "object",
            "anyOf": [facet_presentation, canonical_value],
            "description": "Either vertices and facets or an unchanged canonical FiniteSimplicialComplex value.",
        }


def simplicial_complex_request_from_value(
    value: FiniteSimplicialComplex,
) -> SimplicialComplexRequest:
    """Explicitly project a canonical complex to the facet request carrier."""

    return SimplicialComplexRequest(
        vertices=value.vertices,
        facets=value.maximal_simplices,
    )


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
            raise _validation_error(
                "topology.require_canonical_faces_1",
                "faces must use canonical vertex order and dimension",
            )
        if tuple(sorted(set(self.faces))) != self.faces:
            raise _validation_error(
                "topology.require_canonical_faces_2",
                "faces must be unique and lexicographically ordered",
            )
        return self


class FiniteSimplicialComplex(StrictModel):
    """Canonical non-empty faces of one finite abstract simplicial complex."""

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

    @model_validator(mode="after")
    def require_complete_canonical_complex(self) -> Self:
        if tuple(sorted(set(self.vertices))) != self.vertices:
            raise _validation_error(
                "topology.require_complete_canonical_complex_1",
                "complex vertices must be unique and canonical",
            )
        canonical_facets = _require_request_complex(
            self.vertices, self.maximal_simplices, check_closure=False
        )
        if canonical_facets != self.maximal_simplices:
            raise _validation_error(
                "topology.require_complete_canonical_complex_2",
                "maximal simplices must be canonical",
            )
        expected_f_vector = tuple(len(item.faces) for item in self.faces_by_dimension)
        if (
            self.dimension != len(self.faces_by_dimension) - 1
            or self.f_vector != expected_f_vector
            or self.closure_size != sum(self.f_vector)
        ):
            raise _validation_error(
                "topology.require_complete_canonical_complex_4",
                "complex dimension, f-vector, or closure size shape is invalid",
            )
        return self


class SimplicialComplexCanonicalizationResult(StrictModel):
    complex: FiniteSimplicialComplex


def require_linear_algebra_bounds(complex_: FiniteSimplicialComplex) -> None:
    sizes = complex_.f_vector
    if any(size > MAX_TOPOLOGY_CHAIN_GROUP for size in sizes):
        raise _validation_error(
            "topology.require_linear_algebra_bounds_1",
            f"each chain group may contain at most {MAX_TOPOLOGY_CHAIN_GROUP} faces",
        )
    padded = (0, *sizes)
    if any(
        rows * columns > MAX_TOPOLOGY_MATRIX_CELLS for rows, columns in pairwise(padded)
    ):
        raise _validation_error(
            "topology.require_linear_algebra_bounds_2",
            f"a boundary matrix exceeds the {MAX_TOPOLOGY_MATRIX_CELLS}-cell bound",
        )


def _require_canonical_conversion_bounds(
    complex_: FiniteSimplicialComplex,
    convention: HomologyConvention,
) -> None:
    """Every simplicial chain producer must fit the canonical value's bounds.

    ``ChainComplexValue`` caps the aggregate cells across every
    differential, so admission must check the same sum rather than each
    boundary product separately.
    """
    from jacobian.math.topology.chain_complexes.values import (
        MAX_BASIS_SIZE,
        MAX_MATRIX_CELLS,
    )

    sizes = (
        (1, *complex_.f_vector)
        if convention is HomologyConvention.REDUCED
        else complex_.f_vector
    )
    if any(size > MAX_BASIS_SIZE for size in sizes):
        raise _validation_error(
            "topology.require_canonical_conversion_bounds_1",
            "simplicial chain complexes require at most "
            f"{MAX_BASIS_SIZE} faces per chain group",
        )
    total_cells = sum(rows * columns for rows, columns in pairwise(sizes))
    if total_cells > MAX_MATRIX_CELLS:
        raise _validation_error(
            "topology.require_canonical_conversion_bounds_2",
            "simplicial chain complexes require "
            f"{total_cells} aggregate boundary cells within the canonical "
            f"{MAX_MATRIX_CELLS}-cell bound",
        )


class ChainComplexRequest(StrictModel):
    complex: FiniteSimplicialComplex
    coefficient_ring: ChainCoefficientRing = ChainCoefficientRing.INTEGER
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED


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
            raise _validation_error(
                "topology.require_canonical_sparse_entries_1",
                "sparse entries must be unique and row-major",
            )
        if any(
            entry.row >= self.rows or entry.column >= self.columns or entry.value == 0
            for entry in self.entries
        ):
            raise _validation_error(
                "topology.require_canonical_sparse_entries_2",
                "sparse entry lies outside the matrix or stores zero",
            )
        return self


class BoundarySquareLedgerEntry(StrictModel):
    upper_dimension: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_DIMENSION)
    product_rows: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    product_columns: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    nonzero_entries: Literal[0] = 0


def _resolve_chain_coefficient_values(
    coefficient_ring: ChainCoefficientRing,
    prime: StrictInt | None,
) -> set[int]:
    if coefficient_ring is ChainCoefficientRing.INTEGER:
        if prime is not None:
            raise _validation_error(
                "topology.resolve_chain_coefficient_values_1",
                "integer result must not declare a prime",
            )
        return {-1, 1}
    if prime is None or not is_bounded_prime(prime):
        raise _validation_error(
            "topology.resolve_chain_coefficient_values_2",
            "prime-field result requires a bounded prime",
        )
    return set(range(1, prime))


def _validate_chain_convention_augmentation(
    convention: HomologyConvention,
    augmentation: SparseBoundaryMatrix | None,
) -> None:
    if convention is HomologyConvention.REDUCED:
        if augmentation is None:
            raise _validation_error(
                "topology.validate_chain_convention_augmentation_1",
                "reduced chains require the augmentation map",
            )
    elif augmentation is not None:
        raise _validation_error(
            "topology.validate_chain_convention_augmentation_2",
            "unreduced chains must not include an augmentation",
        )


class ChainComplexResult(StrictModel):
    complex: FiniteSimplicialComplex
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
    canonical_value: ChainComplexValue

    @model_validator(mode="after")
    def require_coherent_chain_contract(self) -> Self:
        allowed_values = _resolve_chain_coefficient_values(
            self.coefficient_ring, self.prime
        )
        dimensions = tuple(item.dimension for item in self.simplex_bases)
        if dimensions != tuple(range(len(self.simplex_bases))):
            raise _validation_error(
                "topology.require_coherent_chain_contract_1",
                "simplex bases must cover contiguous dimensions",
            )
        if tuple(matrix.source_dimension for matrix in self.boundary_matrices) != (
            dimensions
        ):
            raise _validation_error(
                "topology.require_coherent_chain_contract_2",
                "boundary matrices must align with simplex bases",
            )
        for matrix in self.boundary_matrices:
            if any(entry.value not in allowed_values for entry in matrix.entries):
                raise _validation_error(
                    "topology.require_coherent_chain_contract_3",
                    "boundary coefficient is outside its coefficient ring",
                )
        _validate_chain_convention_augmentation(self.convention, self.augmentation)
        expected_ledger = tuple(range(1, len(self.simplex_bases)))
        if tuple(item.upper_dimension for item in self.boundary_squared_zero) != (
            expected_ledger
        ):
            raise _validation_error(
                "topology.require_coherent_chain_contract_4",
                "boundary-square ledger must cover every adjacent pair",
            )
        canonical_ring = (
            CoefficientRing.INTEGER
            if self.coefficient_ring is ChainCoefficientRing.INTEGER
            else CoefficientRing.PRIME_FIELD
        )
        expected_basis_sizes = tuple(len(item.simplices) for item in self.simplex_bases)
        expected_basis_sizes = (
            (1, *expected_basis_sizes)
            if self.convention is HomologyConvention.REDUCED
            else expected_basis_sizes
        )
        expected_degree_min = -1 if self.convention is HomologyConvention.REDUCED else 0
        if (
            self.canonical_value.coefficient_ring is not canonical_ring
            or self.canonical_value.prime != self.prime
            or self.canonical_value.degree_min != expected_degree_min
            or self.canonical_value.basis_sizes != expected_basis_sizes
        ):
            raise _validation_error(
                "topology.require_coherent_chain_contract_5",
                "canonical chain-complex value does not match the simplicial coefficient, convention, or basis axes",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the chain kernel established all derived fields."""

        return cls.model_construct(**values)


__all__ = [
    "MAX_BARYCENTRIC_SOURCE_FACES",
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
    "Simplex",
    "SimplexBasis",
    "SimplicialComplexCanonicalizationResult",
    "SimplicialComplexRequest",
    "SparseBoundaryMatrix",
    "SparseMatrixEntry",
    "VertexLabel",
    "face_closure",
    "simplicial_complex_request_from_value",
]


class BarycentricSubdivisionRequest(StrictModel):
    """Subdivide a complex via its order complex (barycentric subdivision)."""

    complex: SimplicialComplexRequest


class BarycentricSubdivisionResult(StrictModel):
    """The barycentric subdivision as a facet list."""

    original_vertices: tuple[str, ...]
    original_dimension: int
    subdivision_vertices: tuple[str, ...]
    subdivision_facets: tuple[tuple[str, ...], ...]
    num_new_vertices: int
    complex: FiniteSimplicialComplex
    subdivision_complex: FiniteSimplicialComplex | None = None
    subdivision_vertex_faces: tuple[tuple[str, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_structural_subdivision(self) -> Self:
        if self.num_new_vertices != len(self.subdivision_vertices):
            raise _validation_error(
                "topology.require_subdivision_canonical_1",
                "num_new_vertices must match subdivision_vertices",
            )
        if not self.subdivision_facets:
            if self.subdivision_complex is not None:
                raise _validation_error(
                    "topology.require_subdivision_canonical_2",
                    "empty subdivision must have no complex",
                )
        else:
            if self.subdivision_complex is None:
                raise _validation_error(
                    "topology.require_subdivision_canonical_3",
                    "non-empty subdivision requires subdivision_complex",
                )
            if tuple(sorted(self.subdivision_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.subdivision_facets)
            ):
                raise _validation_error(
                    "topology.require_subdivision_canonical_4",
                    "subdivision_complex maximal simplices must match subdivision_facets",
                )
            if tuple(sorted(self.subdivision_complex.vertices)) != tuple(
                sorted(self.subdivision_vertices)
            ):
                raise _validation_error(
                    "topology.require_subdivision_canonical_5",
                    "subdivision_complex vertices must match subdivision_vertices",
                )
            # Validate vertex labels are bounded and injective
            if len(set(self.subdivision_vertices)) != len(self.subdivision_vertices):
                raise _validation_error(
                    "topology.require_subdivision_canonical_6",
                    "subdivision vertices must be unique",
                )
            for label in self.subdivision_vertices:
                if len(label) > 32 or not label[0].isalnum():
                    raise _validation_error(
                        "topology.require_subdivision_canonical_7",
                        f"invalid subdivision vertex label: {label}",
                    )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the admitted subdivision kernel established the result."""

        return cls.model_construct(**values)


class ShellingCheckRequest(StrictModel):
    """Check a submitted shelling order of a pure complex."""

    complex: SimplicialComplexRequest
    facet_order: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )


class ShellingCheckResult(StrictModel):
    """Result of checking a shelling order, bound to its checked source."""

    complex: FiniteSimplicialComplex
    facet_order: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )
    is_shelling: bool
    failed_at: int | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def require_structural_shelling(self) -> Self:
        if self.is_shelling:
            if self.failed_at is not None or self.failure_reason is not None:
                raise _validation_error(
                    "topology.require_structural_shelling_1",
                    "a shelling result cannot carry failure diagnostics",
                )
        elif (
            self.failed_at is None
            or not 0 <= self.failed_at < len(self.facet_order)
            or not self.failure_reason
        ):
            raise _validation_error(
                "topology.require_structural_shelling_2",
                "a failed shelling result requires a valid position and reason",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the admitted shelling kernel established the decision."""

        return cls.model_construct(**values)


__all__.extend(
    [
        "BarycentricSubdivisionRequest",
        "BarycentricSubdivisionResult",
        "ShellingCheckRequest",
        "ShellingCheckResult",
    ]
)
