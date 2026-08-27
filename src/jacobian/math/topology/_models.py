"""Bounded contracts for exact finite simplicial topology."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    StrictInt,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import canonicalize_json
from jacobian.math.chain_complexes.values import ChainComplexValue

MAX_TOPOLOGY_VERTICES = 64
MAX_TOPOLOGY_FACETS = 128
MAX_TOPOLOGY_DIMENSION = 7
MAX_TOPOLOGY_FACES = 2048
MAX_TOPOLOGY_CHAIN_GROUP = 512
MAX_TOPOLOGY_MATRIX_CELLS = 131_072
MAX_TOPOLOGY_PRIME = 251


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

    faces: list[set[Simplex]] = [set() for _ in range(MAX_TOPOLOGY_DIMENSION + 1)]
    for facet in facets:
        for size in range(1, len(facet) + 1):
            faces[size - 1].update(combinations(facet, size))
    highest = max(index for index, values in enumerate(faces) if values)
    return tuple(tuple(sorted(values)) for values in faces[: highest + 1])


def canonical_complex(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> FiniteSimplicialComplex:
    """Construct the neutral canonical value for validated facet data."""

    canonical_vertices = tuple(sorted(vertices))
    canonical_facets = tuple(sorted(tuple(sorted(facet)) for facet in facets))
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
        complex_digest=simplicial_complex_digest(
            vertices=canonical_vertices,
            maximal_simplices=canonical_facets,
            faces_by_dimension=faces_by_dimension,
            dimension=dimension,
            f_vector=f_vector,
            closure_size=closure_size,
        ),
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
    closure = face_closure(tuple(canonical))
    if sum(map(len, closure)) > MAX_TOPOLOGY_FACES:
        raise _validation_error(
            "topology.require_request_complex_7",
            f"face closure may contain at most {MAX_TOPOLOGY_FACES} non-empty faces",
        )
    return tuple(sorted(canonical))


_CANONICAL_COMPLEX_DUMP_KEYS = frozenset(
    {
        "vertices",
        "maximal_simplices",
        "faces_by_dimension",
        "dimension",
        "f_vector",
        "closure_size",
        "orientation_convention",
        "empty_simplex_stored",
        "complex_digest",
    }
)


class SimplicialComplexRequest(StrictModel):
    """A bounded facet presentation for canonicalization.

    Accepts either the facet presentation (``vertices`` + ``facets``) or an
    unchanged canonical :class:`FiniteSimplicialComplex` value, so results
    such as ``VertexDeletionResult.remaining_complex`` can feed structural
    requests directly.
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
        # A before-validator switches the remaining validation to python
        # semantics, where strict tuples reject JSON arrays; normalize the
        # accepted array shapes to tuples so strict JSON dispatch (the only
        # transport path) keeps admitting facet presentations.
        if isinstance(data, FiniteSimplicialComplex):
            return {
                "vertices": tuple(data.vertices),
                "facets": tuple(tuple(facet) for facet in data.maximal_simplices),
            }
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        data = normalized
        if "facets" in data and "maximal_simplices" in data:
            raise _validation_error(
                "topology.accept_canonical_complex_value_1",
                "pass either facets or a canonical FiniteSimplicialComplex "
                "value, not both",
            )
        if "maximal_simplices" not in data:
            return data
        unknown = sorted(set(data) - _CANONICAL_COMPLEX_DUMP_KEYS)
        if unknown:
            raise _validation_error(
                "topology.accept_canonical_complex_value_2",
                f"unknown fields alongside maximal_simplices: {unknown}",
            )
        if "vertices" not in data:
            raise _validation_error(
                "topology.accept_canonical_complex_value_3",
                "canonical complex value must carry vertices",
            )
        try:
            canonical = FiniteSimplicialComplex.model_validate(data)
        except ValidationError as error:
            raise _validation_error(
                "topology.accept_canonical_complex_value_4",
                "canonical complex value must be a valid "
                f"FiniteSimplicialComplex dump: {error.error_count()} "
                "check(s) failed",
            ) from error
        return {
            "vertices": tuple(canonical.vertices),
            "facets": tuple(tuple(facet) for facet in canonical.maximal_simplices),
        }

    @model_validator(mode="after")
    def require_bounded_maximal_facets(self) -> Self:
        if any(
            not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1 for facet in self.facets
        ):
            raise _validation_error(
                "topology.require_bounded_maximal_facets_1",
                "each facet must contain between 1 and "
                f"{MAX_TOPOLOGY_DIMENSION + 1} vertices",
            )
        _require_request_complex(self.vertices, self.facets)
        return self

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> JsonSchemaValue:
        """Advertise both accepted input shapes in the published schema.

        The runtime accepts either the facet presentation or an unchanged
        canonical ``FiniteSimplicialComplex`` dump (the before-validator
        normalizes the latter), so schema-guided callers must see both
        alternatives; otherwise a serialized canonical value could not be
        validated against any advertised consumer schema.
        """
        facet_presentation = handler(core_schema)
        canonical_value = handler(FiniteSimplicialComplex.__pydantic_core_schema__)
        facet_presentation.pop("title", None)
        return {
            "anyOf": [facet_presentation, canonical_value],
            "description": (
                "Either the facet presentation (vertices + facets) or an "
                "unchanged canonical FiniteSimplicialComplex value."
            ),
        }


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
            raise _validation_error(
                "topology.require_digest_binds_canonical_complex_1",
                "complex_digest does not bind the canonical complex",
            )
        return value

    @model_validator(mode="after")
    def require_complete_canonical_complex(self) -> Self:
        if tuple(sorted(set(self.vertices))) != self.vertices:
            raise _validation_error(
                "topology.require_complete_canonical_complex_1",
                "complex vertices must be unique and canonical",
            )
        canonical_facets = _require_request_complex(
            self.vertices,
            self.maximal_simplices,
        )
        if canonical_facets != self.maximal_simplices:
            raise _validation_error(
                "topology.require_complete_canonical_complex_2",
                "maximal simplices must be canonical",
            )
        closure = face_closure(self.maximal_simplices)
        expected_faces = tuple(
            FacesInDimension(dimension=dimension, faces=faces)
            for dimension, faces in enumerate(closure)
        )
        if self.faces_by_dimension != expected_faces:
            raise _validation_error(
                "topology.require_complete_canonical_complex_3",
                "faces_by_dimension is not the complete face closure",
            )
        expected_f_vector = tuple(len(faces) for faces in closure)
        if (
            self.dimension != len(closure) - 1
            or self.f_vector != expected_f_vector
            or self.closure_size != sum(expected_f_vector)
        ):
            raise _validation_error(
                "topology.require_complete_canonical_complex_4",
                "complex dimension, f-vector, or closure size is invalid",
            )
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


def _require_canonical_conversion_bounds(complex_: FiniteSimplicialComplex) -> None:
    """Unreduced GF(p) chains must fit the canonical value's bounds.

    ``ChainComplexValue`` caps the aggregate cells across every
    differential, so admission must check the same sum rather than each
    boundary product separately.
    """
    from jacobian.math.chain_complexes.values import (
        MAX_BASIS_SIZE,
        MAX_MATRIX_CELLS,
    )

    sizes = complex_.f_vector
    if any(size > MAX_BASIS_SIZE for size in sizes):
        raise _validation_error(
            "topology.require_canonical_conversion_bounds_1",
            "unreduced prime-field chain complexes require at most "
            f"{MAX_BASIS_SIZE} faces per chain group",
        )
    padded = (0, *sizes)
    total_cells = sum(rows * columns for rows, columns in pairwise(padded))
    if total_cells > MAX_MATRIX_CELLS:
        raise _validation_error(
            "topology.require_canonical_conversion_bounds_2",
            "unreduced prime-field chain complexes require "
            f"{total_cells} aggregate boundary cells within the canonical "
            f"{MAX_MATRIX_CELLS}-cell bound",
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
                raise _validation_error(
                    "topology.require_coefficient_semantics_and_bounds_1",
                    "integer chain complexes must not declare a prime",
                )
        elif self.prime is None or not is_bounded_prime(self.prime):
            raise _validation_error(
                "topology.require_coefficient_semantics_and_bounds_2",
                "prime-field chain complexes require a bounded prime",
            )
        require_linear_algebra_bounds(self.complex)
        # Every accepted unreduced prime-field producer result must carry
        # its canonical chain-complex value, whose basis and cell bounds
        # are tighter than the sparse internal ones.
        if (
            self.coefficient_ring is ChainCoefficientRing.PRIME_FIELD
            and self.convention is HomologyConvention.UNREDUCED
        ):
            _require_canonical_conversion_bounds(self.complex)
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
    product_is_zero: Literal[True] = True


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
    canonical_value: ChainComplexValue | None = None

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
        if self.canonical_value is not None and not (
            self.coefficient_ring is ChainCoefficientRing.PRIME_FIELD
            and self.convention is HomologyConvention.UNREDUCED
        ):
            raise _validation_error(
                "topology.require_coherent_chain_contract_5",
                "canonical chain-complex value is only defined for unreduced "
                "prime-field chains",
            )
        if self.canonical_value is not None:
            from jacobian.math.topology._chain_conversion import (
                canonical_chain_complex_value_from_parts,
            )

            expected = canonical_chain_complex_value_from_parts(
                self.coefficient_ring,
                self.convention,
                self.prime,
                self.simplex_bases,
                self.boundary_matrices,
            )
            if self.canonical_value != expected:
                raise _validation_error(
                    "topology.require_coherent_chain_contract_6",
                    "canonical chain-complex value must match retained chain data",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the chain kernel established all derived fields."""

        return cls(**values)


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
    "Simplex",
    "SimplexBasis",
    "SimplicialComplexCanonicalizationResult",
    "SimplicialComplexRequest",
    "SparseBoundaryMatrix",
    "SparseMatrixEntry",
    "VertexLabel",
    "face_closure",
    "simplicial_complex_digest",
]


class BarycentricSubdivisionRequest(StrictModel):
    """Subdivide a complex via its order complex (barycentric subdivision)."""

    complex: SimplicialComplexRequest

    @model_validator(mode="after")
    def require_barycentric_work_bounds(self) -> Self:
        closure = face_closure(tuple(tuple(sorted(f)) for f in self.complex.facets))
        face_count = sum(len(part) for part in closure)
        if face_count > 31:
            raise _validation_error(
                "topology.require_barycentric_work_bounds_1",
                f"barycentric subdivision requires at most 31 faces (got {face_count}); "
                "input would produce >128 subdivision facets exceeding result contract",
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
    complex: SimplicialComplexRequest
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

        return cls(**values)


class ShellingCheckRequest(StrictModel):
    """Check a submitted shelling order of a pure complex."""

    complex: SimplicialComplexRequest
    facet_order: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_FACETS,
    )

    @model_validator(mode="after")
    def require_valid_order(self) -> Self:
        if sorted(self.facet_order) != list(range(len(self.complex.facets))):
            raise _validation_error(
                "topology.require_valid_order_1",
                "facet_order must be a permutation of facet indices",
            )
        return self


class ShellingCheckResult(TopologyExactResult):
    """Result of checking a shelling order, bound to its checked source."""

    complex: SimplicialComplexRequest
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

        return cls(**values)


__all__.extend(
    [
        "BarycentricSubdivisionRequest",
        "BarycentricSubdivisionResult",
        "ShellingCheckRequest",
        "ShellingCheckResult",
    ]
)
