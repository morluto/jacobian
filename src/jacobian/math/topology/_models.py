"""Bounded contracts for exact finite simplicial topology."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Any, Literal, NamedTuple, Self

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


def _all_nonempty_faces(facets: tuple[Simplex, ...]) -> set[Simplex]:
    faces: set[Simplex] = set()
    for facet in facets:
        canonical = tuple(sorted(facet))
        for size in range(1, len(canonical) + 1):
            faces.update(combinations(canonical, size))
    return faces


def _maximal_faces(faces: Iterable[Simplex]) -> tuple[tuple[str, ...], ...]:
    """Extract maximal faces in canonical (-size, face) order."""

    maximal: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in sorted(faces, key=lambda f: (-len(f), f)):
        face_set = frozenset(face)
        if not any(existing.issuperset(face_set) for existing in seen):
            maximal.append(tuple(sorted(face)))
            seen.add(face_set)
    return tuple(maximal)


def _join_maximal_facets(
    facets_a: tuple[Simplex, ...],
    facets_b: tuple[Simplex, ...],
) -> tuple[tuple[str, ...], ...]:
    return _maximal_faces(
        tuple(sorted(set(fa) | set(fb))) for fa in facets_a for fb in facets_b
    )


def _skeleton_maximal_facets(
    facets: tuple[Simplex, ...],
    k: int,
) -> tuple[tuple[str, ...], ...]:
    faces = {
        face
        for facet in facets
        for face in combinations(sorted(facet), min(k + 1, len(facet)))
    }
    return _maximal_faces(faces)


def _collapse_remaining_facets(
    facets: tuple[Simplex, ...],
    free_face: tuple[str, ...],
    coface_face: tuple[str, ...],
) -> tuple[tuple[str, ...], ...] | None:
    """Mirror the elementary-collapse interval removal.

    Returns the remaining maximal facets when the collapse applies (the free
    face is contained in exactly one facet equal to the coface) and ``None``
    when the operation would report ``is_free_face=False`` instead.
    """

    free_set = frozenset(free_face)
    coface_set = frozenset(coface_face)
    containing = [
        frozenset(facet) for facet in facets if free_set.issubset(frozenset(facet))
    ]
    if len(containing) != 1 or containing[0] != coface_set:
        return None
    remaining = {
        face
        for face in _all_nonempty_faces(facets)
        if not (free_set.issubset(set(face)) and set(face).issubset(coface_set))
    }
    return _maximal_faces(remaining)


def _shelling_restriction_failure(
    facet_i: set[str],
    prev_facets: list[set[str]],
) -> str | None:
    """Check the standard shelling restriction for one new facet.

    ``R(F_i)`` — the faces of ``F_i`` contained in no earlier facet — must be
    a nonempty interval of the face lattice, i.e. have exactly one minimal
    face. Returns the failure description or ``None`` when the restriction is
    admissible.
    """

    faces_not_in_prev: set[tuple[str, ...]] = set()
    n = len(facet_i)
    for r in range(1, n + 1):
        for subset in combinations(sorted(facet_i), r):
            face_set = set(subset)
            if not any(face_set.issubset(pf) for pf in prev_facets):
                faces_not_in_prev.add(subset)
    if not faces_not_in_prev:
        return "has no new faces"
    face_sets = {frozenset(face) for face in faces_not_in_prev}
    min_faces = [
        face
        for face in face_sets
        if not any(other < face for other in face_sets if other != face)
    ]
    if len(min_faces) != 1:
        return "restriction is not an interval"
    return None


def _evaluate_shelling(
    facets: tuple[Simplex, ...],
    facet_order: tuple[int, ...],
) -> tuple[bool, int | None, str | None]:
    """Recompute the shelling decision for a submitted facet order."""

    dim = len(facets[0])
    if not all(len(facet) == dim for facet in facets):
        return False, 0, "complex is not pure"

    for i, idx in enumerate(order := facet_order):
        facet_i = set(facets[idx])
        if i == 0:
            continue
        prev_facets = [set(facets[order[j]]) for j in range(i)]
        for j in range(i):
            intersection = facet_i & set(facets[order[j]])
            if intersection == facet_i:
                return (
                    False,
                    i,
                    f"facet {idx} is contained in earlier facet",
                )
        failure = _shelling_restriction_failure(facet_i, prev_facets)
        if failure is not None:
            return False, i, f"facet {idx} {failure}"

    return True, None, None


class _PseudomanifoldExpectation(NamedTuple):
    """Recomputed pseudomanifold decision for one facet family."""

    is_pseudomanifold: bool
    is_closed: bool
    obstruction: str | None


def _all_faces(facets: tuple[Simplex, ...]) -> set[tuple[str, ...]]:
    """Return the complete set of nonempty faces for a facet list."""
    faces: set[tuple[str, ...]] = set()
    for facet in facets:
        n = len(facet)
        for r in range(1, n + 1):
            for subset in combinations(facet, r):
                faces.add(tuple(sorted(subset)))
    return faces


def _cover_relations(face_frozens: list[frozenset[str]]) -> list[list[int]]:
    """Compute the cover relation of the face lattice: i covers j when
    ``face_frozens[i] < face_frozens[j]`` with no strict intermediate."""

    n = len(face_frozens)
    covers: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if face_frozens[i] < face_frozens[j]:
                has_intermediate = any(
                    face_frozens[i] < face_frozens[k] < face_frozens[j]
                    for k in range(n)
                )
                if not has_intermediate:
                    covers[i].append(j)
    return covers


def _minimal_face_indices(face_frozens: list[frozenset[str]]) -> list[int]:
    is_minimal = [True] * len(face_frozens)
    for i in range(len(face_frozens)):
        for j in range(len(face_frozens)):
            if face_frozens[j] < face_frozens[i]:
                is_minimal[i] = False
                break
    return [i for i, minimal in enumerate(is_minimal) if minimal]


def _maximal_chains_from_covers(
    covers: list[list[int]],
    minimal_indices: list[int],
    n: int,
    sorted_faces: list[tuple[str, ...]],
    face_frozens: list[frozenset[str]],
) -> list[list[int]]:
    maximal_chains: list[list[int]] = []

    def dfs(chain: list[int]) -> None:
        last = chain[-1]
        if not covers[last]:
            maximal_chains.append(list(chain))
            return
        for nxt in covers[last]:
            chain.append(nxt)
            dfs(chain)
            chain.pop()

    for start in minimal_indices:
        dfs([start])
    # If no minimal (should not happen) fallback to each face as chain
    if not maximal_chains and sorted_faces:
        # Single-face complex
        for i in range(n):
            if not any(face_frozens[i] < face_frozens[j] for j in range(n)):
                # maximal element
                maximal_chains.append([i])
    return maximal_chains


def _replayed_barycentric_subdivision(
    facets: tuple[Simplex, ...],
    faces: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Recompute the barycentric-subdivision facets for ``bv{i}`` labels
    assigned to ``faces`` in the given canonical order."""

    face_frozens = [frozenset(face) for face in faces]
    covers = _cover_relations(face_frozens)
    minimal_indices = _minimal_face_indices(face_frozens)
    maximal_chains = _maximal_chains_from_covers(
        covers, minimal_indices, len(faces), faces, face_frozens
    )
    vertex_map = {face: f"bv{idx}" for idx, face in enumerate(faces)}
    subdivision_facets = [
        tuple(sorted(vertex_map[faces[idx]] for idx in chain))
        for chain in maximal_chains
    ]
    return tuple(sorted(set(subdivision_facets), key=lambda f: (-len(f), f)))


def _require_subdivision_replay(
    *,
    source_complex: SimplicialComplexRequest,
    original_vertices: tuple[str, ...],
    subdivision_vertices: tuple[str, ...],
    subdivision_vertex_faces: tuple[tuple[str, ...], ...],
    num_new_vertices: int,
    subdivision_facets: tuple[tuple[str, ...], ...],
) -> None:
    """Replay the deterministic subdivision against the retained source so
    every derived field — including the indexed vertex-to-face map — must
    equal what the kernel produces for this exact labeling.  In particular,
    swapping entries of ``subdivision_vertex_faces`` while keeping the
    subdivision complex unchanged cannot validate."""

    faces = sorted(_all_faces(source_complex.facets), key=lambda f: (len(f), f))
    expected_labels = [f"bv{i}" for i in range(len(faces))]
    if (
        list(subdivision_vertices) != expected_labels
        or list(subdivision_vertex_faces) != faces
        or num_new_vertices != len(faces)
    ):
        raise ValueError(
            "subdivision_vertices and subdivision_vertex_faces must be "
            "the canonical indexed bijection onto the source complex's "
            "non-empty faces"
        )
    expected = _replayed_barycentric_subdivision(source_complex.facets, faces)
    if (
        tuple(subdivision_facets) != expected
        or original_vertices != source_complex.vertices
    ):
        raise ValueError("subdivision facets do not match the retained source complex")


def _codim1_incidence(
    facets: list[frozenset[str]],
) -> dict[frozenset[str], int]:
    counts: dict[frozenset[str], int] = {}
    for facet in facets:
        for face in combinations(sorted(facet), len(facet) - 1):
            key = frozenset(face)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _expected_pseudomanifold_decision(
    facets: list[frozenset[str]],
    dim: int,
) -> _PseudomanifoldExpectation:
    """Recompute the bounded pseudomanifold decision from incidence."""

    is_pure = all(len(facet) - 1 == dim for facet in facets) if facets else False
    if not is_pure:
        return _PseudomanifoldExpectation(
            False, False, "not pure: facets have different dimensions"
        )
    # Dimension-zero complexes participate in the same incidence definition:
    # each vertex facet contains exactly one codimension-one face, the empty
    # face, so one point has it once (boundary) and two points twice (closed).
    counts = _codim1_incidence(facets)
    if any(count > 2 for count in counts.values()):
        for face, count in counts.items():
            if count > 2:
                return _PseudomanifoldExpectation(
                    False,
                    False,
                    f"codim-1 face {sorted(face)} is in {count} facets",
                )
    is_closed = all(count == 2 for count in counts.values()) if counts else False
    return _PseudomanifoldExpectation(
        True, is_closed, None if is_closed else "pseudomanifold with boundary"
    )


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


_CANONICAL_COMPLEX_DUMP_KEYS = frozenset(
    {
        "complex_format",
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
        vertices = normalized.get("vertices")
        if isinstance(vertices, list):
            normalized["vertices"] = tuple(vertices)
        facets = normalized.get("facets")
        if isinstance(facets, list):
            normalized["facets"] = tuple(
                tuple(facet) if isinstance(facet, list) else facet for facet in facets
            )
        data = normalized
        if "facets" in data and "maximal_simplices" in data:
            raise ValueError(
                "pass either facets or a canonical FiniteSimplicialComplex "
                "value, not both"
            )
        if "maximal_simplices" not in data:
            return data
        unknown = sorted(set(data) - _CANONICAL_COMPLEX_DUMP_KEYS)
        if unknown:
            raise ValueError(f"unknown fields alongside maximal_simplices: {unknown}")
        if "vertices" not in data:
            raise ValueError("canonical complex value must carry vertices")
        try:
            canonical = FiniteSimplicialComplex.model_validate(data)
        except ValidationError as error:
            raise ValueError(
                "canonical complex value must be a valid "
                f"FiniteSimplicialComplex dump: {error.error_count()} "
                "check(s) failed"
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
            raise ValueError(
                "each facet must contain between 1 and "
                f"{MAX_TOPOLOGY_DIMENSION + 1} vertices"
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
    simplex: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

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
    simplex: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

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
    """The closed star of a simplex, bound to its source complex."""

    complex: SimplicialComplexRequest
    simplex: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    star_facets: tuple[tuple[str, ...], ...]
    star_is_empty: bool
    star_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_star_binding(self) -> Self:
        target = frozenset(self.simplex)
        if len(target) != len(self.simplex):
            raise ValueError("star simplex vertices must be distinct")
        # Replay the star relation over the retained source so an authored
        # simplex or facet list cannot validate independently of it.
        if not any(target.issubset(frozenset(facet)) for facet in self.complex.facets):
            raise ValueError("simplex must be a face of the retained complex")
        expected_facets = tuple(
            tuple(sorted(facet))
            for facet in sorted(
                (
                    frozenset(facet)
                    for facet in self.complex.facets
                    if target.issubset(facet)
                ),
                key=lambda value: (-len(value), sorted(value)),
            )
        )
        if self.star_is_empty != (not expected_facets):
            raise ValueError("star_is_empty does not match the source replay")
        if tuple(self.star_facets) != expected_facets:
            raise ValueError("star_facets do not match the source-complex replay")
        if self.star_is_empty:
            if self.star_complex is not None:
                raise ValueError("empty star must have no complex")
        else:
            if self.star_complex is None:
                raise ValueError("non-empty star requires star_complex")
            if tuple(sorted(self.star_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(f)) for f in self.star_facets)
            ):
                raise ValueError(
                    "star_complex maximal simplices must match star_facets"
                )
            if set(self.star_complex.vertices) != {
                v for facet in self.star_facets for v in facet
            }:
                raise ValueError("star_complex vertices must match star_facets")
        return self


class VertexDeletionRequest(StrictModel):
    """Delete a vertex subset from a simplicial complex.

    The deletion must leave at least one simplex on the remaining vertices;
    deleting every vertex is rejected because the empty complex has no
    canonical value.
    """

    complex: SimplicialComplexRequest
    vertices_to_delete: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
        description=(
            "Vertex subset to remove. The deletion must leave at least one "
            "simplex on the remaining vertices: deleting every vertex is "
            "rejected because the empty complex has no canonical value."
        ),
    )

    @model_validator(mode="after")
    def require_valid_deletion(self) -> Self:
        vtd = set(self.vertices_to_delete)
        if len(vtd) != len(self.vertices_to_delete):
            raise ValueError("vertices_to_delete must be distinct")
        if not vtd.issubset(set(self.complex.vertices)):
            raise ValueError("vertices_to_delete must be in the complex")
        # The canonical complex value cannot represent the empty complex, so
        # a deletion whose induced subcomplex would be empty is out of
        # contract and must be rejected at the boundary.
        if all(frozenset(facet).issubset(vtd) for facet in self.complex.facets):
            raise ValueError(
                "deletion must leave at least one simplex on the remaining vertices"
            )
        return self


class VertexDeletionResult(TopologyExactResult):
    """The induced subcomplex after deleting a vertex subset."""

    complex: SimplicialComplexRequest
    deleted_vertices: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
    )
    remaining_vertices: tuple[str, ...]
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_complex: FiniteSimplicialComplex

    @model_validator(mode="after")
    def require_deletion_canonical(self) -> Self:
        # Replay the induced-subcomplex relation against the retained source
        # so a serialized result cannot validate independently of it.
        deleted = set(self.deleted_vertices)
        if len(deleted) != len(self.deleted_vertices):
            raise ValueError("deleted_vertices must be distinct")
        if not deleted.issubset(set(self.complex.vertices)):
            raise ValueError("deleted_vertices must be in the retained complex")
        if tuple(self.deleted_vertices) != tuple(sorted(self.deleted_vertices)):
            raise ValueError("deleted_vertices must use canonical vertex order")
        remaining_faces = [
            face
            for face in _all_nonempty_faces(
                tuple(tuple(sorted(f)) for f in self.complex.facets)
            )
            if not (set(face) & deleted)
        ]
        expected_facets = _maximal_faces(remaining_faces)
        expected_vertices = tuple(
            sorted({v for facet in expected_facets for v in facet})
        )
        if tuple(self.remaining_facets) != expected_facets:
            raise ValueError(
                "remaining_facets must be the induced subcomplex of the "
                "retained source complex"
            )
        if tuple(self.remaining_vertices) != expected_vertices:
            raise ValueError("remaining_vertices must match the source-complex replay")
        if tuple(sorted(self.remaining_complex.maximal_simplices)) != tuple(
            sorted(tuple(sorted(f)) for f in self.remaining_facets)
        ):
            raise ValueError(
                "remaining_complex maximal simplices must match remaining_facets"
            )
        if tuple(sorted(self.remaining_complex.vertices)) != tuple(
            sorted(self.remaining_vertices)
        ):
            raise ValueError("remaining_complex vertices must match remaining_vertices")
        return self


class SkeletonRequest(StrictModel):
    """Request the k-skeleton of a simplicial complex."""

    complex: SimplicialComplexRequest
    k: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)

    @model_validator(mode="after")
    def require_admissible_skeleton(self) -> Self:
        # Admit the request only when the exact k-skeleton satisfies every
        # canonical result bound before execution (e.g. the 3-skeleton of
        # eight disjoint 7-simplices yields 560 tetrahedra > 128 facets).
        skeleton_facets = _skeleton_maximal_facets(self.complex.facets, self.k)
        skeleton_vertices = tuple(
            sorted({v for facet in skeleton_facets for v in facet})
        )
        SimplicialComplexRequest(vertices=skeleton_vertices, facets=skeleton_facets)
        return self


class SkeletonResult(TopologyExactResult):
    """The k-skeleton as a facet list, bound to its source complex."""

    complex: SimplicialComplexRequest
    k: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    skeleton_facets: tuple[tuple[str, ...], ...]
    skeleton_vertices: tuple[str, ...]
    skeleton_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_skeleton_canonical(self) -> Self:
        # Replay the exact k-skeleton against the retained source so the
        # returned complex cannot disagree with k (e.g. an edge in a
        # 0-skeleton) or with the source itself.
        expected_facets = _skeleton_maximal_facets(self.complex.facets, self.k)
        if tuple(self.skeleton_facets) != expected_facets:
            raise ValueError(
                "skeleton_facets must be the exact k-skeleton of the "
                "retained source complex"
            )
        expected_vertices = tuple(
            sorted({v for facet in expected_facets for v in facet})
        )
        if tuple(self.skeleton_vertices) != expected_vertices:
            raise ValueError("skeleton_vertices must match the k-skeleton replay")
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


def _require_join_admission(
    complex_a: SimplicialComplexRequest,
    complex_b: SimplicialComplexRequest,
) -> None:
    """Shared join admission checked BEFORE any facet-product expansion.

    Disjoint maximal operand facets pair into pairwise-distinct maximal
    unions, so the exact join facet count is the product below and every
    derived bound (vertices, facet width, facet count) is checkable
    without expanding that product or running the quadratic maximal-face
    scan.
    """
    va = set(complex_a.vertices)
    vb = set(complex_b.vertices)
    if va & vb:
        raise ValueError("join requires disjoint vertex sets; rename vertices first")
    combined_vertices = tuple(sorted(va | vb))
    if len(combined_vertices) > MAX_TOPOLOGY_VERTICES:
        raise ValueError(
            f"join would span {len(combined_vertices)} vertices, above "
            f"the {MAX_TOPOLOGY_VERTICES}-vertex canonical bound"
        )
    widest_join_facet = max(len(facet) for facet in complex_a.facets) + max(
        len(facet) for facet in complex_b.facets
    )
    if widest_join_facet > MAX_TOPOLOGY_DIMENSION + 1:
        raise ValueError(
            f"join facets would span {widest_join_facet} vertices, "
            f"above the {MAX_TOPOLOGY_DIMENSION + 1}-vertex facet bound"
        )
    join_facet_count = len(complex_a.facets) * len(complex_b.facets)
    if join_facet_count > MAX_TOPOLOGY_FACETS:
        raise ValueError(
            f"join would carry {join_facet_count} maximal facets, above "
            f"the {MAX_TOPOLOGY_FACETS}-facet result contract"
        )


class JoinRequest(StrictModel):
    """Join two simplicial complexes on disjoint vertex sets."""

    complex_a: SimplicialComplexRequest
    complex_b: SimplicialComplexRequest

    @model_validator(mode="after")
    def require_admissible_join(self) -> Self:
        _require_join_admission(self.complex_a, self.complex_b)
        combined_vertices = tuple(
            sorted(set(self.complex_a.vertices) | set(self.complex_b.vertices))
        )
        join_facets = _join_maximal_facets(self.complex_a.facets, self.complex_b.facets)
        SimplicialComplexRequest(vertices=combined_vertices, facets=join_facets)
        return self


def _require_complex_matches_facets(
    complex_value: FiniteSimplicialComplex | None,
    *,
    facets: tuple[tuple[str, ...], ...],
    vertices: tuple[str, ...],
    empty_message: str,
    missing_message: str,
    facets_message: str,
    vertices_message: str,
) -> None:
    """Require an optional canonical complex value to restate a facet list."""
    if not facets:
        if complex_value is not None:
            raise ValueError(empty_message)
        return
    if complex_value is None:
        raise ValueError(missing_message)
    if tuple(sorted(complex_value.maximal_simplices)) != tuple(
        sorted(tuple(sorted(f)) for f in facets)
    ):
        raise ValueError(facets_message)
    if tuple(sorted(complex_value.vertices)) != tuple(sorted(vertices)):
        raise ValueError(vertices_message)


class JoinResult(TopologyExactResult):
    """The join of two complexes, bound to both operands."""

    complex_a: SimplicialComplexRequest
    complex_b: SimplicialComplexRequest
    join_vertices: tuple[str, ...]
    join_facets: tuple[tuple[str, ...], ...]
    join_dimension: int
    join_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_join_canonical(self) -> Self:
        # Replay the facet union against the retained operands so any
        # internally canonical complex cannot pass as "the" join result.
        # Apply join admission FIRST so an oversized serialized operand
        # pair is rejected without repeating the expensive expansion.
        _require_join_admission(self.complex_a, self.complex_b)
        vertices_a = set(self.complex_a.vertices)
        vertices_b = set(self.complex_b.vertices)
        expected_facets = _join_maximal_facets(
            self.complex_a.facets, self.complex_b.facets
        )
        if tuple(self.join_facets) != expected_facets:
            raise ValueError(
                "join_facets must be the exact facet union of the retained operands"
            )
        expected_vertices = tuple(sorted(vertices_a | vertices_b))
        if tuple(self.join_vertices) != expected_vertices:
            raise ValueError("join_vertices must match the operand vertex sets")
        expected_dimension = (
            max(len(facet) - 1 for facet in expected_facets) if expected_facets else 0
        )
        if self.join_dimension != expected_dimension:
            raise ValueError("join_dimension must match the replayed facet union")
        _require_complex_matches_facets(
            self.join_complex,
            facets=self.join_facets,
            vertices=self.join_vertices,
            empty_message="empty join must have no complex",
            missing_message="non-empty join requires join_complex",
            facets_message="join_complex maximal simplices must match join_facets",
            vertices_message="join_complex vertices must match join_vertices",
        )
        if self.join_complex is not None and (
            self.join_complex.dimension != self.join_dimension
        ):
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
    complex: SimplicialComplexRequest
    subdivision_complex: FiniteSimplicialComplex | None = None
    subdivision_vertex_faces: tuple[tuple[str, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_subdivision_canonical(self) -> Self:
        # Replay must satisfy the subdivision request's work admission, so a
        # serialized result cannot bypass the bounded source domain (a source
        # whose exact subdivision exceeds the result contract is impossible
        # to obtain from the operation).
        BarycentricSubdivisionRequest(complex=self.complex)
        # The advertised original dimension must equal the dimension derived
        # from the retained source complex's facets.
        source_dimension = max(len(facet) for facet in self.complex.facets) - 1
        if self.original_dimension != source_dimension:
            raise ValueError(
                "original_dimension must match the retained source complex"
            )
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
            # Validate vertex labels are bounded and injective
            if len(set(self.subdivision_vertices)) != len(self.subdivision_vertices):
                raise ValueError("subdivision vertices must be unique")
            for label in self.subdivision_vertices:
                if len(label) > 32 or not label[0].isalnum():
                    raise ValueError(f"invalid subdivision vertex label: {label}")
        _require_subdivision_replay(
            source_complex=self.complex,
            original_vertices=self.original_vertices,
            subdivision_vertices=self.subdivision_vertices,
            subdivision_vertex_faces=self.subdivision_vertex_faces,
            num_new_vertices=self.num_new_vertices,
            subdivision_facets=self.subdivision_facets,
        )
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
        expected = _expected_pseudomanifold_decision(facets, dim)
        if self.is_pseudomanifold != expected.is_pseudomanifold:
            raise ValueError(
                f"is_pseudomanifold {self.is_pseudomanifold} does not match "
                f"expected {expected.is_pseudomanifold}"
            )
        if not expected.is_pseudomanifold:
            if self.is_closed:
                raise ValueError("non-pseudomanifold cannot be closed")
            if self.obstruction != expected.obstruction:
                raise ValueError(
                    f"obstruction {self.obstruction!r} does not match replayed "
                    f"{expected.obstruction!r}"
                )
            return self
        if self.is_closed != expected.is_closed:
            raise ValueError("is_closed must match codim-1 incidence")
        if self.obstruction != expected.obstruction:
            raise ValueError(
                f"obstruction {self.obstruction!r} does not match expected "
                f"{expected.obstruction!r}"
            )
        return self


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
            raise ValueError("facet_order must be a permutation of facet indices")
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
    def require_shelling_binding(self) -> Self:
        # Apply the request model's permutation requirement before replay: a
        # partial or duplicated order visits only its own facets and must not
        # authenticate a decision about facets it omits.
        if sorted(self.facet_order) != list(range(len(self.complex.facets))):
            raise ValueError("facet_order must be a permutation of facet indices")
        # Replay the shelling condition over the retained complex and order so
        # an authored decision cannot validate independently of its source.
        expected_is_shelling, expected_failed_at, expected_reason = _evaluate_shelling(
            self.complex.facets, self.facet_order
        )
        if self.is_shelling != expected_is_shelling:
            raise ValueError(
                f"is_shelling {self.is_shelling} does not match replayed "
                f"decision {expected_is_shelling}"
            )
        if self.failed_at != expected_failed_at:
            raise ValueError(
                f"failed_at {self.failed_at} does not match replayed "
                f"{expected_failed_at}"
            )
        if self.failure_reason != expected_reason:
            raise ValueError(
                f"failure_reason {self.failure_reason!r} does not match "
                f"replayed {expected_reason!r}"
            )
        return self


class ElementaryCollapseRequest(StrictModel):
    """Check and perform one elementary collapse step (free face must be codimension-one)."""

    complex: SimplicialComplexRequest
    # A coface is a facet, so it carries at most MAX_TOPOLOGY_DIMENSION + 1
    # vertices; capping the candidate tuples keeps accepted request work tied
    # to the complex's bounds even when labels are not vertices at all.
    free_face: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION,
    )
    coface: tuple[VertexLabel, ...] = Field(
        min_length=2,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_valid_collapse(self) -> Self:
        # Repeated labels do not represent simplices; validate uniqueness
        # before converting either field to a set.
        if len(set(self.free_face)) != len(self.free_face):
            raise ValueError(
                "free_face vertices must be unique; repeated labels do not "
                "represent a simplex"
            )
        if len(set(self.coface)) != len(self.coface):
            raise ValueError(
                "coface vertices must be unique; repeated labels do not "
                "represent a simplex"
            )
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
        # Admit the request only when the exact residual complex satisfies the
        # canonical result bounds before execution (e.g. collapsing one face
        # of a 7-simplex beside 127 edges would leave 134 maximal facets).
        remaining_facets = _collapse_remaining_facets(
            self.complex.facets,
            tuple(sorted(self.free_face)),
            tuple(sorted(self.coface)),
        )
        if remaining_facets is not None:
            remaining_vertices = tuple(
                sorted({v for facet in remaining_facets for v in facet})
            )
            if not remaining_vertices:
                return self
            SimplicialComplexRequest(
                vertices=remaining_vertices, facets=remaining_facets
            )
        return self


def _require_collapse_complex(
    remaining_facets: tuple[tuple[str, ...], ...],
    remaining_vertices: tuple[str, ...],
    remaining_complex: FiniteSimplicialComplex | None,
) -> None:
    """Require ``remaining_complex`` to restate the residual facet list."""

    if not remaining_facets:
        if remaining_complex is not None:
            raise ValueError("empty collapsed complex must have no remaining_complex")
        return
    if remaining_complex is None:
        raise ValueError("non-empty collapse requires remaining_complex")
    if tuple(sorted(remaining_complex.maximal_simplices)) != tuple(
        sorted(tuple(sorted(f)) for f in remaining_facets)
    ):
        raise ValueError(
            "remaining_complex maximal simplices must match remaining_facets"
        )
    if tuple(sorted(remaining_complex.vertices)) != tuple(sorted(remaining_vertices)):
        raise ValueError("remaining_complex vertices must match remaining_vertices")


class ElementaryCollapseResult(TopologyExactResult):
    """Result of checking one elementary collapse step, bound to its source."""

    complex: SimplicialComplexRequest
    is_free_face: bool
    free_face: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION,
    )
    coface: tuple[VertexLabel, ...] = Field(
        min_length=2,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_vertices: tuple[str, ...]
    remaining_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_collapse_binding(self) -> Self:
        free_face = tuple(sorted(self.free_face))
        coface = tuple(sorted(self.coface))
        if self.free_face != free_face or self.coface != coface:
            raise ValueError("free_face and coface must use canonical vertex order")
        if (
            len(set(free_face)) != len(free_face)
            or len(set(coface)) != len(coface)
            or len(coface) != len(free_face) + 1
            or not set(free_face).issubset(coface)
        ):
            raise ValueError("free_face must be codimension-one in coface")
        # Replay freeness and the residual construction against the retained
        # source complex so a serialized decision cannot validate on its own.
        replayed_facets = _collapse_remaining_facets(
            self.complex.facets, free_face, coface
        )
        if replayed_facets is None:
            if self.is_free_face:
                raise ValueError(
                    "is_free_face does not match the retained source complex"
                )
            # The kernel echoes a rejected collapse in canonical vertex order
            # (each facet sorted, vertices from the facet union), so the
            # expected artifacts must be canonicalized the same way before
            # comparison; raw source order would make the operation reject
            # its own typed result for a noncanonical presentation.
            expected_facets = tuple(
                tuple(sorted(facet)) for facet in self.complex.facets
            )
            expected_vertices = tuple(
                sorted({vertex for facet in self.complex.facets for vertex in facet})
            )
        else:
            if not self.is_free_face:
                raise ValueError(
                    "is_free_face does not match the retained source complex"
                )
            expected_facets = replayed_facets
            expected_vertices = tuple(
                sorted({vertex for facet in replayed_facets for vertex in facet})
            )
        if tuple(sorted(self.remaining_facets)) != tuple(sorted(expected_facets)):
            raise ValueError("remaining_facets do not match the source-complex replay")
        if tuple(sorted(self.remaining_vertices)) != tuple(sorted(expected_vertices)):
            raise ValueError(
                "remaining_vertices do not match the source-complex replay"
            )
        _require_collapse_complex(
            self.remaining_facets,
            self.remaining_vertices,
            self.remaining_complex,
        )
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
