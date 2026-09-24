"""Local simplicial-transform contracts and deterministic kernels.

This module owns the contracts whose postconditions are local transforms of a
finite simplicial complex.  The canonical complex and chain values deliberately
remain in :mod:`_models`; homology, shelling, barycentric subdivision, and
pseudomanifolds retain their already separate owners.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, pairwise
from math import comb
from typing import Any, Self

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FiniteSimplicialComplex,
    Simplex,
    SimplicialComplexRequest,
    VertexLabel,
    _require_request_complex,
    _validation_error,
    canonical_complex,
    face_closure,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
    require_complex_admission,
    run_topology_admission,
)

MAX_FACE_ENUMERATOR_CANDIDATES = MAX_TOPOLOGY_FACETS * (
    (1 << (MAX_TOPOLOGY_DIMENSION + 1)) - 1
)

MAX_MINIMAL_NONFACE_CANDIDATES = 1 << 14
MAX_MINIMAL_NONFACE_WORK = 430_000
MAX_MINIMAL_NONFACES = 4_096
MAX_MINIMAL_NONFACE_RESULT_BYTES = 2_500_000
MAX_STANLEY_REISNER_GENERATORS = 64
MAX_STANLEY_REISNER_RESULT_BYTES = 2_500_000
MAX_VERTEX_LABEL_BYTES = 32
MAX_CANONICAL_COMPLEX_JSON_BYTES = (
    (MAX_TOPOLOGY_FACES + MAX_TOPOLOGY_FACETS)
    * (MAX_TOPOLOGY_DIMENSION + 1)
    * (MAX_VERTEX_LABEL_BYTES + 3)
    + MAX_TOPOLOGY_VERTICES * (MAX_VERTEX_LABEL_BYTES + 3)
    + 8_192
)


def _all_nonempty_faces(facets: tuple[Simplex, ...]) -> set[Simplex]:
    faces: set[Simplex] = set()
    for facet in facets:
        canonical = tuple(sorted(facet))
        for size in range(1, len(canonical) + 1):
            faces.update(combinations(canonical, size))
    return faces


def _bounded_face_closure(
    facets: tuple[Simplex, ...], *, max_faces: int
) -> tuple[tuple[Simplex, ...], ...]:
    """Build a face closure while enforcing its unique-face cap incrementally."""
    faces_by_dimension: list[set[Simplex]] = [
        set() for _ in range(MAX_TOPOLOGY_DIMENSION + 1)
    ]
    face_count = 0
    for facet in facets:
        canonical = tuple(sorted(facet))
        for size in range(1, len(canonical) + 1):
            dimension_faces = faces_by_dimension[size - 1]
            for face in combinations(canonical, size):
                if face in dimension_faces:
                    continue
                if face_count == max_faces:
                    raise OperationResourceAdmissionError(
                        location=("complex",),
                        code="topology.face_enumerator.admission.output_faces",
                        message=(
                            "the face closure exceeds the "
                            f"{max_faces}-face output envelope"
                        ),
                    )
                dimension_faces.add(face)
                face_count += 1
    populated = [index for index, values in enumerate(faces_by_dimension) if values]
    if not populated:
        return ()
    highest = max(populated)
    return tuple(tuple(sorted(values)) for values in faces_by_dimension[: highest + 1])


def _maximal_faces(faces: Iterable[Simplex]) -> tuple[tuple[str, ...], ...]:
    """Extract maximal faces in canonical ``(-size, face)`` order."""

    maximal: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for face in sorted(faces, key=lambda value: (-len(value), value)):
        face_set = frozenset(face)
        if not any(existing.issuperset(face_set) for existing in seen):
            maximal.append(tuple(sorted(face)))
            seen.add(face_set)
    return tuple(maximal)


def join_maximal_facets(
    facets_a: tuple[Simplex, ...], facets_b: tuple[Simplex, ...]
) -> tuple[tuple[str, ...], ...]:
    if not facets_a:
        return facets_b
    if not facets_b:
        return facets_a
    return _maximal_faces(
        tuple(sorted(set(facet_a) | set(facet_b)))
        for facet_a in facets_a
        for facet_b in facets_b
    )


def skeleton_maximal_facets(
    facets: tuple[Simplex, ...], k: int
) -> tuple[tuple[str, ...], ...]:
    return _maximal_faces(
        face
        for facet in facets
        for face in combinations(sorted(facet), min(k + 1, len(facet)))
    )


def collapse_remaining_facets(
    facets: tuple[Simplex, ...], free_face: tuple[str, ...], coface: tuple[str, ...]
) -> tuple[tuple[str, ...], ...] | None:
    """Return the residual facets, or ``None`` when the face is not free."""

    free_set = frozenset(free_face)
    coface_set = frozenset(coface)
    containing = [
        frozenset(facet) for facet in facets if free_set.issubset(frozenset(facet))
    ]
    if len(containing) != 1 or containing[0] != coface_set:
        return None
    return _maximal_faces(
        face
        for face in _all_nonempty_faces(facets)
        if not (free_set.issubset(face) and set(face).issubset(coface_set))
    )


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
    if not facets:
        if complex_value is not None and (
            complex_value.vertices
            or complex_value.maximal_simplices
            or complex_value.faces_by_dimension
            or complex_value.dimension != -1
        ):
            raise _validation_error(
                "topology.require_complex_matches_facets_1",
                "empty result must use the canonical empty complex",
            )
        return
    if complex_value is None:
        raise _validation_error(
            "topology.require_complex_matches_facets_2", missing_message
        )
    if tuple(sorted(complex_value.maximal_simplices)) != tuple(
        sorted(tuple(sorted(facet)) for facet in facets)
    ):
        raise _validation_error(
            "topology.require_complex_matches_facets_3", facets_message
        )
    if tuple(sorted(complex_value.vertices)) != tuple(sorted(vertices)):
        raise _validation_error(
            "topology.require_complex_matches_facets_4", vertices_message
        )


def _require_simplex_in_complex(
    complex_: SimplicialComplexRequest, simplex: tuple[str, ...]
) -> None:
    simplex_set = set(simplex)
    if len(simplex_set) != len(simplex):
        raise ValueError("simplex vertices must be distinct")
    if not simplex_set.issubset(complex_.vertices):
        raise ValueError("simplex vertices must be in the complex")
    if not any(simplex_set.issubset(facet) for facet in complex_.facets):
        raise ValueError("simplex must be a face of the complex")


def _require_deletion(
    complex_: SimplicialComplexRequest, vertices_to_delete: tuple[str, ...]
) -> None:
    deleted = set(vertices_to_delete)
    if len(deleted) != len(vertices_to_delete):
        raise ValueError("vertices_to_delete must be distinct")
    if not deleted.issubset(complex_.vertices):
        raise ValueError("vertices_to_delete must be in the complex")


def _require_collapse(
    complex_: SimplicialComplexRequest,
    free_face: tuple[str, ...],
    coface: tuple[str, ...],
) -> None:
    free_set, coface_set = set(free_face), set(coface)
    if len(free_set) != len(free_face) or len(coface_set) != len(coface):
        raise ValueError("collapse faces must have distinct vertices")
    if not free_set < coface_set:
        raise ValueError("free_face must be a proper subset of coface")
    if len(coface_set) != len(free_set) + 1:
        raise ValueError("elementary collapse requires codimension-one faces")


class FVectorRequest(StrictModel):
    """Request the f-vector and h-vector of a simplicial complex."""

    complex: SimplicialComplexRequest


class FaceEnumeratorRequest(StrictModel):
    """Request the polynomial counting all faces by cardinality."""

    complex: SimplicialComplexRequest


class MinimalNonfacesRequest(StrictModel):
    """Request the inclusion-minimal nonfaces of one canonical complex."""

    complex: FiniteSimplicialComplex = Field(
        description=(
            "Canonical finite complex on at most 14 vertices; powerset and "
            "result-byte admission use the full source vertex domain."
        )
    )


class MinimalNonfacesResult(StrictModel):
    """Source-bound canonical antichain of minimal nonfaces."""

    source: FiniteSimplicialComplex
    minimal_nonfaces: tuple[Simplex, ...] = Field(max_length=MAX_MINIMAL_NONFACES)

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteSimplicialComplex,
        minimal_nonfaces: tuple[Simplex, ...],
    ) -> Self:
        """Build an admitted exact antichain without replaying its search."""
        return cls.model_construct(source=source, minimal_nonfaces=minimal_nonfaces)


class VertexVariableBinding(StrictModel):
    """One source vertex and its collision-free polynomial variable."""

    vertex: VertexLabel
    variable: PolynomialVariable


class StanleyReisnerIdealRequest(StrictModel):
    """Construct the squarefree monomial ideal of one finite complex."""

    complex: FiniteSimplicialComplex


class StanleyReisnerIdealResult(StrictModel):
    """The polynomial ideal together with its exact ordered vertex binding."""

    source: FiniteSimplicialComplex
    ideal: RationalPolynomialIdeal
    vertex_variables: tuple[VertexVariableBinding, ...] = Field(
        max_length=MAX_POLYNOMIAL_VARIABLES
    )


class FVectorResult(StrictModel):
    """The f-vector ``(f_-1, f_0, ..., f_d)`` and its h-vector."""

    f_vector: tuple[int, ...]
    h_vector: tuple[int, ...]
    euler_characteristic: int
    dimension: int


class InducedSubcomplexRequest(StrictModel):
    """Select a vertex subset and take its induced subcomplex."""

    complex: FiniteSimplicialComplex
    selected_vertices: tuple[VertexLabel, ...] = Field(
        min_length=0,
        max_length=MAX_TOPOLOGY_VERTICES,
        description="Distinct vertices to retain; an empty selection returns {∅}.",
    )


class InducedFaceImage(StrictModel):
    source_face: Simplex
    induced_face: Simplex | None


class InducedSubcomplexResult(StrictModel):
    complex: FiniteSimplicialComplex
    selected_vertices: tuple[VertexLabel, ...]
    induced_complex: FiniteSimplicialComplex
    face_images: tuple[InducedFaceImage, ...] = Field(max_length=MAX_TOPOLOGY_FACES)

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class GVectorResult(StrictModel):
    """The initial h-differences through the conventional midpoint."""

    f_vector: tuple[int, ...]
    h_vector: tuple[int, ...]
    g_vector: tuple[int, ...]
    dimension: int


class LinkRequest(StrictModel):
    """Request the link of a simplex in a simplicial complex."""

    complex: SimplicialComplexRequest
    simplex: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )


class LinkResult(StrictModel):
    """The maximal facets of the link of a simplex."""

    simplex: tuple[str, ...]
    link_facets: tuple[tuple[str, ...], ...]
    link_is_empty: bool


class StarRequest(StrictModel):
    """Request the closed star of a simplex in a simplicial complex."""

    complex: SimplicialComplexRequest
    simplex: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )


class StarResult(StrictModel):
    """The closed star produced for a simplex."""

    complex: SimplicialComplexRequest
    simplex: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    star_facets: tuple[tuple[str, ...], ...]
    star_is_empty: bool
    star_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_structural_star(self) -> Self:
        if len(set(self.simplex)) != len(self.simplex):
            raise _validation_error(
                "topology.require_star_binding_1",
                "star simplex vertices must be distinct",
            )
        if self.star_is_empty != (not self.star_facets):
            raise _validation_error(
                "topology.require_star_binding_2",
                "star_is_empty must match whether star_facets is empty",
            )
        if self.star_is_empty:
            if self.star_complex is not None:
                raise _validation_error(
                    "topology.require_star_binding_3", "empty star must have no complex"
                )
        else:
            if self.star_complex is None:
                raise _validation_error(
                    "topology.require_star_binding_4",
                    "non-empty star requires star_complex",
                )
            if tuple(sorted(self.star_complex.maximal_simplices)) != tuple(
                sorted(tuple(sorted(facet)) for facet in self.star_facets)
            ):
                raise _validation_error(
                    "topology.require_star_binding_5",
                    "star_complex maximal simplices must match star_facets",
                )
            if set(self.star_complex.vertices) != {
                vertex for facet in self.star_facets for vertex in facet
            }:
                raise _validation_error(
                    "topology.require_star_binding_6",
                    "star_complex vertices must match star_facets",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class VertexDeletionRequest(StrictModel):
    """Delete a vertex subset from a simplicial complex.

    Deleting all vertices returns the zero-vertex complex ``{∅}``.
    """

    complex: SimplicialComplexRequest
    vertices_to_delete: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
        description="Vertex subset to remove; deleting all vertices returns {∅}.",
    )


class VertexDeletionResult(StrictModel):
    """The induced subcomplex produced after deleting a vertex subset."""

    complex: SimplicialComplexRequest
    deleted_vertices: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_VERTICES
    )
    remaining_vertices: tuple[str, ...]
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_complex: FiniteSimplicialComplex

    @model_validator(mode="after")
    def require_structural_deletion(self) -> Self:
        deleted = set(self.deleted_vertices)
        if len(deleted) != len(self.deleted_vertices):
            raise _validation_error(
                "topology.require_deletion_canonical_1",
                "deleted_vertices must be distinct",
            )
        if tuple(self.deleted_vertices) != tuple(sorted(self.deleted_vertices)):
            raise _validation_error(
                "topology.require_deletion_canonical_3",
                "deleted_vertices must use canonical vertex order",
            )
        if tuple(sorted(self.remaining_complex.maximal_simplices)) != tuple(
            sorted(tuple(sorted(facet)) for facet in self.remaining_facets)
        ):
            raise _validation_error(
                "topology.require_deletion_canonical_4",
                "remaining_complex maximal simplices must match remaining_facets",
            )
        if tuple(sorted(self.remaining_complex.vertices)) != tuple(
            sorted(self.remaining_vertices)
        ):
            raise _validation_error(
                "topology.require_deletion_canonical_5",
                "remaining_complex vertices must match remaining_vertices",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class SkeletonRequest(StrictModel):
    """Request the k-skeleton of a simplicial complex."""

    complex: SimplicialComplexRequest
    k: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)


class SkeletonResult(StrictModel):
    """The k-skeleton as a facet list."""

    complex: SimplicialComplexRequest
    k: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    skeleton_facets: tuple[tuple[str, ...], ...]
    skeleton_vertices: tuple[str, ...]
    skeleton_complex: FiniteSimplicialComplex

    @model_validator(mode="after")
    def require_structural_skeleton(self) -> Self:
        if tuple(sorted(self.skeleton_complex.maximal_simplices)) != tuple(
            sorted(tuple(sorted(facet)) for facet in self.skeleton_facets)
        ):
            raise _validation_error(
                "topology.require_skeleton_canonical_3",
                "skeleton_complex maximal simplices must match skeleton_facets",
            )
        if tuple(sorted(self.skeleton_complex.vertices)) != tuple(
            sorted(self.skeleton_vertices)
        ):
            raise _validation_error(
                "topology.require_skeleton_canonical_4",
                "skeleton_complex vertices must match skeleton_vertices",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _require_join_admission(
    complex_a: SimplicialComplexRequest, complex_b: SimplicialComplexRequest
) -> None:
    vertices_a, vertices_b = set(complex_a.vertices), set(complex_b.vertices)
    if vertices_a & vertices_b:
        raise _validation_error(
            "topology.require_join_admission_1",
            "join requires disjoint vertex sets; rename vertices first",
        )
    if len(vertices_a | vertices_b) > MAX_TOPOLOGY_VERTICES:
        raise _validation_error(
            "topology.require_join_admission_2",
            f"join would span {len(vertices_a | vertices_b)} vertices, above the {MAX_TOPOLOGY_VERTICES}-vertex canonical bound",
        )
    width = max(map(len, complex_a.facets), default=0) + max(
        map(len, complex_b.facets), default=0
    )
    if width > MAX_TOPOLOGY_DIMENSION + 1:
        raise _validation_error(
            "topology.require_join_admission_3",
            f"join facets would span {width} vertices, above the {MAX_TOPOLOGY_DIMENSION + 1}-vertex facet bound",
        )
    count = len(complex_a.facets) * len(complex_b.facets)
    if count > MAX_TOPOLOGY_FACETS:
        raise _validation_error(
            "topology.require_join_admission_4",
            f"join would carry {count} maximal facets, above the {MAX_TOPOLOGY_FACETS}-facet result contract",
        )


class JoinRequest(StrictModel):
    """Join two simplicial complexes on disjoint vertex sets."""

    complex_a: SimplicialComplexRequest
    complex_b: SimplicialComplexRequest


class JoinResult(StrictModel):
    """The join of two complexes."""

    complex_a: SimplicialComplexRequest
    complex_b: SimplicialComplexRequest
    join_vertices: tuple[str, ...]
    join_facets: tuple[tuple[str, ...], ...]
    join_dimension: int
    join_complex: FiniteSimplicialComplex

    @model_validator(mode="after")
    def require_structural_join(self) -> Self:
        _require_complex_matches_facets(
            self.join_complex,
            facets=self.join_facets,
            vertices=self.join_vertices,
            empty_message="empty join must have no complex",
            missing_message="non-empty join requires join_complex",
            facets_message="join_complex maximal simplices must match join_facets",
            vertices_message="join_complex vertices must match join_vertices",
        )
        if (
            self.join_complex is not None
            and self.join_complex.dimension != self.join_dimension
        ):
            raise _validation_error(
                "topology.require_join_canonical_1",
                "join_complex dimension must match join_dimension",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class ConeFaceTransport(StrictModel):
    """One source face and its cone face obtained by adding the apex."""

    source_face: tuple[str, ...] = Field(min_length=0)
    cone_face: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_structural_transport(self) -> Self:
        if tuple(sorted(self.source_face)) != self.source_face:
            raise _validation_error(
                "topology.require_cone_transport_1",
                "transport source faces must use canonical vertex order",
            )
        if tuple(sorted(self.cone_face)) != self.cone_face:
            raise _validation_error(
                "topology.require_cone_transport_2",
                "transport cone faces must use canonical vertex order",
            )
        if not set(self.source_face) < set(self.cone_face):
            raise _validation_error(
                "topology.require_cone_transport_3",
                "a cone face must strictly contain its source face",
            )
        if len(self.cone_face) != len(self.source_face) + 1:
            raise _validation_error(
                "topology.require_cone_transport_4",
                "a cone face adds exactly one vertex to its source face",
            )
        return self


class ConeRequest(StrictModel):
    """Join one complex with a single fresh tagged apex vertex."""

    complex: SimplicialComplexRequest
    apex: VertexLabel = "cone_apex"


class ConeResult(StrictModel):
    """The complete cone complex with its apex and source-face transport."""

    complex: SimplicialComplexRequest
    apex: str
    cone_vertices: tuple[str, ...]
    cone_facets: tuple[tuple[str, ...], ...]
    cone_dimension: int
    cone_complex: FiniteSimplicialComplex
    source_face_transport: tuple[ConeFaceTransport, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_structural_cone(self) -> Self:
        _require_complex_matches_facets(
            self.cone_complex,
            facets=self.cone_facets,
            vertices=self.cone_vertices,
            empty_message="empty cone must have no complex",
            missing_message="non-empty cone requires cone_complex",
            facets_message="cone_complex maximal simplices must match cone_facets",
            vertices_message="cone_complex vertices must match cone_vertices",
        )
        if self.cone_complex is not None and (
            self.cone_complex.dimension != self.cone_dimension
        ):
            raise _validation_error(
                "topology.require_cone_canonical_1",
                "cone_complex dimension must match cone_dimension",
            )
        if self.apex not in set(self.cone_vertices):
            raise _validation_error(
                "topology.require_cone_canonical_2",
                "the apex must be a vertex of the cone complex",
            )
        for row in self.source_face_transport:
            if set(row.cone_face) != set(row.source_face) | {self.apex}:
                raise _validation_error(
                    "topology.require_cone_canonical_3",
                    "every transported cone face must add exactly the apex",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _require_cone_admission(
    complex_: SimplicialComplexRequest, apex: str
) -> tuple[tuple[str, ...], ...]:
    if apex in set(complex_.vertices):
        raise _validation_error(
            "topology.require_cone_admission_1",
            "cone apex must be a fresh tagged vertex outside the source complex",
        )
    vertices = tuple(sorted(set(complex_.vertices) | {apex}))
    if len(vertices) > MAX_TOPOLOGY_VERTICES:
        raise _validation_error(
            "topology.require_cone_admission_2",
            f"cone would span {len(vertices)} vertices, above the {MAX_TOPOLOGY_VERTICES}-vertex canonical bound",
        )
    facets = (
        tuple(tuple(sorted(set(facet) | {apex})) for facet in complex_.facets)
        if complex_.facets
        else ((apex,),)
    )
    width = max(len(facet) for facet in facets)
    if width > MAX_TOPOLOGY_DIMENSION + 1:
        raise _validation_error(
            "topology.require_cone_admission_3",
            f"cone facets would span {width} vertices, above the {MAX_TOPOLOGY_DIMENSION + 1}-vertex facet bound",
        )
    if len(facets) > MAX_TOPOLOGY_FACETS:
        raise _validation_error(
            "topology.require_cone_admission_4",
            f"cone would carry {len(facets)} maximal facets, above the {MAX_TOPOLOGY_FACETS}-facet result contract",
        )
    return facets


class ElementaryCollapseRequest(StrictModel):
    """Check and perform one elementary collapse step (free face must be codimension-one)."""

    complex: SimplicialComplexRequest
    free_face: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION
    )
    coface: tuple[VertexLabel, ...] = Field(
        min_length=2, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )


class ElementaryCollapseResult(StrictModel):
    """Result of one elementary collapse step."""

    complex: SimplicialComplexRequest
    is_free_face: bool
    free_face: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION
    )
    coface: tuple[VertexLabel, ...] = Field(
        min_length=2, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )
    remaining_facets: tuple[tuple[str, ...], ...]
    remaining_vertices: tuple[str, ...]
    remaining_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_structural_collapse(self) -> Self:
        free_face, coface = tuple(sorted(self.free_face)), tuple(sorted(self.coface))
        if self.free_face != free_face or self.coface != coface:
            raise _validation_error(
                "topology.require_collapse_binding_1",
                "free_face and coface must use canonical vertex order",
            )
        if (
            len(set(free_face)) != len(free_face)
            or len(set(coface)) != len(coface)
            or len(coface) != len(free_face) + 1
            or not set(free_face).issubset(coface)
        ):
            raise _validation_error(
                "topology.require_collapse_binding_2",
                "free_face must be codimension-one in coface",
            )
        _require_complex_matches_facets(
            self.remaining_complex,
            facets=self.remaining_facets,
            vertices=self.remaining_vertices,
            empty_message="empty collapsed complex must have no remaining_complex",
            missing_message="non-empty collapse requires remaining_complex",
            facets_message="remaining_complex maximal simplices must match remaining_facets",
            vertices_message="remaining_complex vertices must match remaining_vertices",
        )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def compute_f_vector(request: FVectorRequest) -> FVectorResult:
    """Compute the exact f-, h-, and Euler vectors of one complex."""
    require_complex_admission(request.complex)

    all_simplices = _all_nonempty_faces(request.complex.facets)
    dimension = max((len(simplex) - 1 for simplex in all_simplices), default=-1)
    face_counts = tuple(
        sum(len(simplex) == degree + 1 for simplex in all_simplices)
        for degree in range(dimension + 1)
    )
    f_vector = (1, *face_counts)
    from math import comb

    ring_dimension = dimension + 1
    h_vector = tuple(
        sum(
            (-1) ** (degree - index)
            * comb(ring_dimension - index, degree - index)
            * f_vector[index]
            for index in range(degree + 1)
        )
        for degree in range(ring_dimension + 1)
    )
    return FVectorResult(
        f_vector=f_vector,
        h_vector=h_vector,
        euler_characteristic=sum(
            (-1) ** degree * count for degree, count in enumerate(face_counts)
        ),
        dimension=dimension,
    )


def compute_face_enumerator(request: FaceEnumeratorRequest) -> IntegerPolynomial:
    """Return ``sum_{face in K} t**|face|`` as a canonical ZZ[x] value.

    The empty face contributes the constant term one. Candidate subface
    generation is admitted from the bounded facet presentation before closure
    expansion; unique output faces are then admitted incrementally as they are
    inserted into the face closure.
    """
    canonical_facets = run_topology_admission(
        lambda: _require_request_complex(
            request.complex.vertices, request.complex.facets, check_closure=False
        ),
        location=("complex",),
    )
    candidates = sum((1 << len(facet)) - 1 for facet in canonical_facets)
    if candidates > MAX_FACE_ENUMERATOR_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.face_enumerator.admission.face_candidates",
            message=(
                f"face-enumerator input admits {candidates} subface candidates, "
                f"above the {MAX_FACE_ENUMERATOR_CANDIDATES}-candidate envelope"
            ),
        )
    closure = _bounded_face_closure(canonical_facets, max_faces=MAX_TOPOLOGY_FACES)
    return IntegerPolynomial(coefficients=(*reversed(tuple(map(len, closure))), 1))


def compute_g_vector(request: FVectorRequest) -> GVectorResult:
    """Compute ``g_0=1`` and ``g_i=h_i-h_(i-1)`` through the midpoint.

    This is only the conventional initial-difference transform. It does not
    assert that the complex is a sphere, a manifold, or that the entries are
    nonnegative.
    """
    result = compute_f_vector(request)
    last_index = (result.dimension + 1) // 2
    g_vector = (
        1,
        *(
            result.h_vector[index] - result.h_vector[index - 1]
            for index in range(1, last_index + 1)
        ),
    )
    return GVectorResult(
        f_vector=result.f_vector,
        h_vector=result.h_vector,
        g_vector=g_vector,
        dimension=result.dimension,
    )


def _minimal_nonface_work(vertex_count: int, facets: tuple[Simplex, ...]) -> int:
    candidate_count = 1 << vertex_count
    candidate_mask_work = 0 if vertex_count == 0 else vertex_count * (1 << (vertex_count - 1))
    immediate_subface_checks = 0 if vertex_count == 0 else vertex_count * (1 << (vertex_count - 1))
    face_candidate_work = sum((1 << len(facet)) - 1 for facet in facets)
    facet_validation_work = (
        len(facets) * (len(facets) - 1) // 2 * (MAX_TOPOLOGY_DIMENSION + 1)
    )
    face_mask_work = MAX_TOPOLOGY_FACES * (MAX_TOPOLOGY_DIMENSION + 1)
    output_label_work = (
        min(
            MAX_MINIMAL_NONFACES,
            comb(vertex_count, vertex_count // 2),
        )
        * vertex_count
    )
    return (
        candidate_count
        + candidate_mask_work
        + immediate_subface_checks
        + face_candidate_work
        + facet_validation_work
        + face_mask_work
        + output_label_work
    )


def _minimal_nonface_result_bytes_bound(source: FiniteSimplicialComplex) -> int:
    vertex_count = len(source.vertices)
    width_bound = min(
        MAX_MINIMAL_NONFACES,
        comb(vertex_count, vertex_count // 2),
    )
    max_row_bytes = (
        sum(len(vertex.encode("ascii")) + 2 for vertex in source.vertices)
        + max(0, vertex_count - 1)
        + 2
    )
    return MAX_CANONICAL_COMPLEX_JSON_BYTES + width_bound * (max_row_bytes + 1) + 2


def _preflight_minimal_nonfaces(
    request: MinimalNonfacesRequest,
) -> FiniteSimplicialComplex:
    if not isinstance(request, MinimalNonfacesRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="topology.minimal_nonfaces.request_type",
            message="request must be a minimal-nonfaces request",
        )
    source = getattr(request, "complex", None)
    if not isinstance(source, FiniteSimplicialComplex):
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.minimal_nonfaces.source_type",
            message="source must be a canonical finite simplicial complex",
        )
    vertices = getattr(source, "vertices", None)
    if type(vertices) is not tuple:
        raise OperationDomainValidationError(
            location=("complex", "vertices"),
            code="topology.minimal_nonfaces.source_shape",
            message="source vertex axis is outside the canonical complex shape",
        )
    vertex_count = len(vertices)
    candidate_count = 1 << vertex_count
    if candidate_count > MAX_MINIMAL_NONFACE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex", "vertices"),
            code="topology.minimal_nonfaces.powerset_over_envelope",
            message=(
                f"the {candidate_count}-subset powerset exceeds the admitted "
                f"{MAX_MINIMAL_NONFACE_CANDIDATES}-candidate envelope"
            ),
        )
    facets = getattr(source, "maximal_simplices", None)
    if (
        type(facets) is not tuple
        or len(facets) > MAX_TOPOLOGY_FACETS
        or (bool(vertices) != bool(facets))
        or any(
            type(facet) is not tuple
            or not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1
            for facet in facets
        )
    ):
        raise OperationDomainValidationError(
            location=("complex", "maximal_simplices"),
            code="topology.minimal_nonfaces.source_shape",
            message="source facets are outside the canonical complex shape",
        )
    estimated_work = _minimal_nonface_work(vertex_count, facets)
    if estimated_work > MAX_MINIMAL_NONFACE_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.minimal_nonfaces.work_over_envelope",
            message=(
                f"minimal-nonface work estimate {estimated_work} exceeds "
                f"the {MAX_MINIMAL_NONFACE_WORK}-step envelope"
            ),
        )
    try:
        validated_source = FiniteSimplicialComplex.model_validate(
            source.model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.minimal_nonfaces.source_invalid",
            message="source is not a valid canonical finite complex",
        ) from exc
    result_bytes_bound = _minimal_nonface_result_bytes_bound(validated_source)
    if result_bytes_bound > MAX_MINIMAL_NONFACE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.minimal_nonfaces.output_over_envelope",
            message=(
                f"source-bound antichain output estimate {result_bytes_bound} "
                f"bytes exceeds the {MAX_MINIMAL_NONFACE_RESULT_BYTES}-byte envelope"
            ),
        )
    return validated_source


def _admitted_nonempty_face_masks(
    source: FiniteSimplicialComplex,
) -> frozenset[int]:
    try:
        closure = face_closure(source.maximal_simplices)
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.minimal_nonfaces.source_invalid",
            message="source facets do not define a canonical finite complex",
        ) from exc
    stored_closure = tuple(tuple(item.faces) for item in source.faces_by_dimension)
    if (
        closure != stored_closure
        or source.f_vector != tuple(map(len, closure))
        or source.closure_size != sum(map(len, closure))
        or sum(map(len, closure)) > MAX_TOPOLOGY_FACES
    ):
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.minimal_nonfaces.source_closure",
            message="source face closure is incomplete or inconsistent",
        )
    vertex_index = {vertex: index for index, vertex in enumerate(source.vertices)}
    return frozenset(
        sum(1 << vertex_index[vertex] for vertex in face)
        for dimension_faces in closure
        for face in dimension_faces
    )


def _minimal_nonface_positions(
    source: FiniteSimplicialComplex,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate minimal nonfaces by their positions on the ordered vertex axis."""

    face_masks = _admitted_nonempty_face_masks(source)
    minimal_nonfaces: list[tuple[int, ...]] = []
    for size in range(2, len(source.vertices) + 1):
        for positions in combinations(range(len(source.vertices)), size):
            mask = sum(1 << position for position in positions)
            if mask not in face_masks and all(
                mask ^ (1 << position) in face_masks for position in positions
            ):
                minimal_nonfaces.append(positions)
    if len(minimal_nonfaces) > MAX_MINIMAL_NONFACES:
        raise OperationResourceAdmissionError(
            location=("minimal_nonfaces",),
            code="topology.minimal_nonfaces.output_over_envelope",
            message=(
                f"antichain has {len(minimal_nonfaces)} members, exceeding "
                f"the {MAX_MINIMAL_NONFACES}-member envelope"
            ),
        )
    return tuple(minimal_nonfaces)


def compute_minimal_nonfaces(
    request: MinimalNonfacesRequest,
) -> MinimalNonfacesResult:
    """Enumerate the canonical minimal-nonface antichain of one complex."""
    source = _preflight_minimal_nonfaces(request)
    minimal_nonfaces = tuple(
        tuple(source.vertices[position] for position in positions)
        for positions in _minimal_nonface_positions(source)
    )
    return MinimalNonfacesResult._from_kernel(
        source=source, minimal_nonfaces=minimal_nonfaces
    )


def compute_stanley_reisner_ideal(
    request: StanleyReisnerIdealRequest,
) -> StanleyReisnerIdealResult:
    """Construct the exact squarefree monomial ideal generated by minimal nonfaces."""

    if not isinstance(request, StanleyReisnerIdealRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="topology.stanley_reisner.request_type",
            message="request must be a Stanley-Reisner ideal request",
        )
    if not isinstance(request.complex, FiniteSimplicialComplex):
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.stanley_reisner.source_type",
            message="source must be a canonical finite simplicial complex",
        )
    if len(request.complex.vertices) > MAX_POLYNOMIAL_VARIABLES:
        raise OperationResourceAdmissionError(
            location=("complex", "vertices"),
            code="topology.stanley_reisner.variable_bound",
            message=(
                "the polynomial ideal carrier admits at most "
                f"{MAX_POLYNOMIAL_VARIABLES} ordered vertex variables"
            ),
        )
    source = _preflight_minimal_nonfaces(
        MinimalNonfacesRequest(complex=request.complex)
    )
    vertex_count = len(source.vertices)
    minimal_nonfaces = _minimal_nonface_positions(source)
    if len(minimal_nonfaces) > MAX_STANLEY_REISNER_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.stanley_reisner.generator_bound",
            message=(
                f"the {len(minimal_nonfaces)} minimal nonfaces exceed the "
                f"{MAX_STANLEY_REISNER_GENERATORS}-generator polynomial ideal bound"
            ),
        )

    # The exact source complex and at most 64 eight-variable sparse terms fit
    # below this conservative canonical JSON envelope before polynomial values
    # are materialized.
    estimated_output_bytes = _minimal_nonface_result_bytes_bound(source) + 64_000
    if estimated_output_bytes > MAX_STANLEY_REISNER_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.stanley_reisner.output_bound",
            message=(
                f"conservative result estimate {estimated_output_bytes} exceeds "
                f"the {MAX_STANLEY_REISNER_RESULT_BYTES}-byte envelope"
            ),
        )

    variables = tuple(f"v{index}" for index in range(vertex_count))
    exponents = tuple(
        sorted(
            (
                tuple(int(index in positions) for index in range(vertex_count))
                for positions in minimal_nonfaces
            ),
            reverse=True,
        )
    )
    one = CanonicalRational(num=1, den=1)
    if exponents:
        generators = tuple(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(RationalPolynomialTerm(coefficient=one, exponents=power),)
                ),
            )
            for power in exponents
        )
    else:
        # The canonical polynomial representation uses the zero polynomial as
        # a singleton generator for the zero ideal, avoiding a second ideal type.
        generators = (
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=()),
            ),
        )
    ideal = RationalPolynomialIdeal(variables=variables, generators=generators)
    bindings = tuple(
        VertexVariableBinding(vertex=vertex, variable=variable)
        for vertex, variable in zip(source.vertices, variables, strict=True)
    )
    return StanleyReisnerIdealResult(
        source=source,
        ideal=ideal,
        vertex_variables=bindings,
    )


def compute_link(request: LinkRequest) -> LinkResult:
    require_complex_admission(request.complex)
    run_topology_admission(
        lambda: _require_simplex_in_complex(request.complex, request.simplex),
        location=("simplex",),
    )
    target = frozenset(request.simplex)
    facets = _maximal_faces(
        tuple(sorted(frozenset(facet) - target))
        for facet in request.complex.facets
        if target.issubset(facet) and frozenset(facet) - target
    )
    return LinkResult(
        simplex=request.simplex, link_facets=facets, link_is_empty=not facets
    )


def compute_star(request: StarRequest) -> StarResult:
    require_complex_admission(request.complex)
    run_topology_admission(
        lambda: _require_simplex_in_complex(request.complex, request.simplex),
        location=("simplex",),
    )
    target = frozenset(request.simplex)
    facets = tuple(
        tuple(sorted(facet))
        for facet in sorted(
            (
                frozenset(facet)
                for facet in request.complex.facets
                if target.issubset(facet)
            ),
            key=lambda value: (-len(value), sorted(value)),
        )
    )
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    return StarResult._from_kernel(
        complex=request.complex,
        simplex=request.simplex,
        star_facets=facets,
        star_is_empty=not facets,
        star_complex=canonical_complex(vertices, facets) if facets else None,
    )


def compute_vertex_deletion(request: VertexDeletionRequest) -> VertexDeletionResult:
    require_complex_admission(request.complex)
    run_topology_admission(
        lambda: _require_deletion(request.complex, request.vertices_to_delete),
        location=("vertices_to_delete",),
    )
    deleted = set(request.vertices_to_delete)
    facets = _maximal_faces(
        face
        for face in _all_nonempty_faces(request.complex.facets)
        if not (set(face) & deleted)
    )
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    return VertexDeletionResult._from_kernel(
        complex=request.complex,
        deleted_vertices=tuple(sorted(deleted)),
        remaining_vertices=vertices,
        remaining_facets=facets,
        remaining_complex=canonical_complex(vertices, facets),
    )


def compute_induced_subcomplex(
    request: InducedSubcomplexRequest,
) -> InducedSubcomplexResult:
    source = request.complex
    run_topology_admission(
        lambda: require_canonical_complex_admission(source), location=("complex",)
    )

    def admit_vertices() -> tuple[str, ...]:
        facet_vertices = tuple(
            sorted({vertex for facet in source.maximal_simplices for vertex in facet})
        )
        if facet_vertices != source.vertices:
            raise ValueError("source vertices do not match its maximal simplices")
        if _maximal_faces(source.maximal_simplices) != source.maximal_simplices:
            raise ValueError("source maximal simplices are not canonical")
        if source.dimension != max(
            (len(facet) - 1 for facet in source.maximal_simplices), default=-1
        ):
            raise ValueError("source dimension does not match its maximal simplices")
        if any(
            item.dimension != dimension
            for dimension, item in enumerate(source.faces_by_dimension)
        ):
            raise ValueError("source face table dimensions are not canonical")
        source_face_count = sum(
            len(dimension.faces) for dimension in source.faces_by_dimension
        )
        if source_face_count != source.closure_size:
            raise ValueError("source closure_size does not match its face table")
        if source_face_count > MAX_TOPOLOGY_FACES:
            raise ValueError("source face closure exceeds the induced-subcomplex bound")
        if len(set(request.selected_vertices)) != len(request.selected_vertices):
            raise ValueError("selected_vertices must be distinct")
        selected = tuple(sorted(request.selected_vertices))
        if not set(selected).issubset(source.vertices):
            raise ValueError("selected_vertices must belong to the source complex")
        return selected

    selected = run_topology_admission(admit_vertices, location=("selected_vertices",))
    selected_set = set(selected)
    source_faces = tuple(
        face for dimension in source.faces_by_dimension for face in dimension.faces
    )
    retained = tuple(face for face in source_faces if set(face).issubset(selected_set))
    facets = _maximal_faces(
        tuple(vertex for vertex in facet if vertex in selected_set)
        for facet in source.maximal_simplices
        if any(vertex in selected_set for vertex in facet)
    )
    groups = tuple(
        tuple(face for face in retained if len(face) == dimension + 1)
        for dimension in range(max(map(len, facets), default=0))
    )
    induced = canonical_complex(selected, facets, closure=groups)
    images = tuple(
        InducedFaceImage(
            source_face=face,
            induced_face=face if set(face).issubset(selected_set) else None,
        )
        for face in source_faces
    )
    return InducedSubcomplexResult._from_kernel(
        complex=source,
        selected_vertices=selected,
        induced_complex=induced,
        face_images=images,
    )


def compute_skeleton(request: SkeletonRequest) -> SkeletonResult:
    require_complex_admission(request.complex)
    facets = skeleton_maximal_facets(request.complex.facets, request.k)
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    if facets:
        require_complex_admission(
            SimplicialComplexRequest(vertices=vertices, facets=facets)
        )
    return SkeletonResult._from_kernel(
        complex=request.complex,
        k=request.k,
        skeleton_facets=facets,
        skeleton_vertices=vertices,
        skeleton_complex=canonical_complex(vertices, facets),
    )


def compute_join(request: JoinRequest) -> JoinResult:
    require_complex_admission(request.complex_a)
    require_complex_admission(request.complex_b)
    run_topology_admission(
        lambda: _require_join_admission(request.complex_a, request.complex_b),
        location=("complex_b",),
    )
    facets = join_maximal_facets(request.complex_a.facets, request.complex_b.facets)
    vertices = tuple(
        sorted(set(request.complex_a.vertices) | set(request.complex_b.vertices))
    )
    dimension = max((len(facet) - 1 for facet in facets), default=-1)
    return JoinResult._from_kernel(
        complex_a=request.complex_a,
        complex_b=request.complex_b,
        join_vertices=vertices,
        join_facets=facets,
        join_dimension=dimension,
        join_complex=canonical_complex(vertices, facets),
    )


def compute_cone(request: ConeRequest) -> ConeResult:
    require_complex_admission(request.complex)
    facets = run_topology_admission(
        lambda: _require_cone_admission(request.complex, request.apex),
        location=("apex",),
    )
    vertices = tuple(sorted(set(request.complex.vertices) | {request.apex}))
    cone_complex = canonical_complex(vertices, facets)
    source_faces = sorted(_all_nonempty_faces(request.complex.facets))
    if not source_faces:
        source_faces = [()]
    transport = tuple(
        ConeFaceTransport(
            source_face=face, cone_face=tuple(sorted(set(face) | {request.apex}))
        )
        for face in source_faces
    )
    return ConeResult._from_kernel(
        complex=request.complex,
        apex=request.apex,
        cone_vertices=vertices,
        cone_facets=tuple(tuple(sorted(facet)) for facet in facets),
        cone_dimension=cone_complex.dimension,
        cone_complex=cone_complex,
        source_face_transport=transport,
    )


class BoundaryRequest(StrictModel):
    """Return the exact boundary subcomplex of a pseudomanifold with boundary."""

    complex: SimplicialComplexRequest


class BoundaryResult(StrictModel):
    """Boundary ridges, their downward closure, and its component count."""

    complex: SimplicialComplexRequest
    dimension: int
    boundary_ridges: tuple[tuple[str, ...], ...]
    boundary_vertices: tuple[str, ...]
    boundary_facets: tuple[tuple[str, ...], ...]
    boundary_complex: FiniteSimplicialComplex
    component_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_structural_boundary(self) -> Self:
        _require_complex_matches_facets(
            self.boundary_complex,
            facets=self.boundary_facets,
            vertices=self.boundary_vertices,
            empty_message="nontrivial boundary must have no empty complex",
            missing_message="nontrivial boundary requires boundary_complex",
            facets_message="boundary_complex maximal simplices must match boundary_facets",
            vertices_message="boundary_complex vertices must match boundary_vertices",
        )
        if tuple(sorted(self.boundary_ridges)) != tuple(
            sorted(tuple(sorted(ridge)) for ridge in self.boundary_facets)
        ):
            raise _validation_error(
                "topology.require_boundary_canonical_1",
                "boundary ridges must match boundary facets",
            )
        if any(len(ridge) != self.dimension for ridge in self.boundary_ridges):
            raise _validation_error(
                "topology.require_boundary_canonical_2",
                "every boundary ridge must be a codimension-one face",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _boundary_ridges(
    facets: tuple[Simplex, ...],
) -> tuple[tuple[str, ...], ...]:
    """Return the sorted codimension-one faces lying in exactly one facet."""

    incidence: dict[frozenset[str], int] = {}
    for facet in facets:
        for face in combinations(sorted(facet), len(facet) - 1):
            key = frozenset(face)
            incidence[key] = incidence.get(key, 0) + 1
    return tuple(
        tuple(sorted(face))
        for face, count in sorted(incidence.items(), key=lambda item: sorted(item[0]))
        if count == 1
    )


def _connected_components(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> int:
    """Count vertex-connected components joined by shared facets."""

    parent = {vertex: vertex for vertex in vertices}

    def find(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for facet in facets:
        for first, second in pairwise(facet):
            parent[find(first)] = find(second)
    return len({find(vertex) for vertex in vertices})


def compute_boundary(request: BoundaryRequest) -> BoundaryResult:
    """Generate the boundary subcomplex of a checked pseudomanifold."""
    from jacobian.math.topology._pseudomanifold import pseudomanifold_decision

    require_complex_admission(request.complex)

    def admit() -> tuple[tuple[str, ...], ...]:
        decision = pseudomanifold_decision(request.complex.facets)
        if not decision.is_pseudomanifold:
            raise _validation_error(
                "topology.require_boundary_admission_1",
                f"boundary requires a pseudomanifold: {decision.obstruction}",
            )
        if decision.is_closed:
            raise _validation_error(
                "topology.require_boundary_admission_2",
                "closed pseudomanifolds have empty boundary; submit a complex "
                "with a boundary ridge",
            )
        if decision.dimension < 1:
            raise _validation_error(
                "topology.require_boundary_admission_3",
                "boundary requires positive dimension",
            )
        ridges = _boundary_ridges(request.complex.facets)
        if not ridges:
            raise _validation_error(
                "topology.require_boundary_admission_4",
                "no codimension-one face has incidence one",
            )
        return ridges

    ridges = run_topology_admission(admit, location=("complex",))
    dimension = max(len(facet) - 1 for facet in request.complex.facets)
    vertices = tuple(sorted({vertex for ridge in ridges for vertex in ridge}))
    return BoundaryResult._from_kernel(
        complex=request.complex,
        dimension=dimension,
        boundary_ridges=ridges,
        boundary_vertices=vertices,
        boundary_facets=ridges,
        boundary_complex=canonical_complex(vertices, ridges),
        component_count=_connected_components(vertices, ridges),
    )


def compute_elementary_collapse(
    request: ElementaryCollapseRequest,
) -> ElementaryCollapseResult:
    require_complex_admission(request.complex)
    run_topology_admission(
        lambda: _require_collapse(request.complex, request.free_face, request.coface),
        location=("free_face", "coface"),
    )
    free_face, coface = tuple(sorted(request.free_face)), tuple(sorted(request.coface))
    facets = collapse_remaining_facets(request.complex.facets, free_face, coface)
    is_free = facets is not None
    if facets is None:
        facets = tuple(tuple(sorted(facet)) for facet in request.complex.facets)
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    return ElementaryCollapseResult._from_kernel(
        complex=request.complex,
        is_free_face=is_free,
        free_face=free_face,
        coface=coface,
        remaining_facets=facets,
        remaining_vertices=vertices,
        remaining_complex=canonical_complex(vertices, facets) if facets else None,
    )


__all__ = [
    "BoundaryRequest",
    "BoundaryResult",
    "ConeFaceTransport",
    "ConeRequest",
    "ConeResult",
    "ElementaryCollapseRequest",
    "ElementaryCollapseResult",
    "FVectorRequest",
    "FVectorResult",
    "FaceEnumeratorRequest",
    "GVectorResult",
    "InducedFaceImage",
    "InducedSubcomplexRequest",
    "InducedSubcomplexResult",
    "JoinRequest",
    "JoinResult",
    "LinkRequest",
    "LinkResult",
    "SkeletonRequest",
    "SkeletonResult",
    "StarRequest",
    "StarResult",
    "VertexDeletionRequest",
    "VertexDeletionResult",
    "collapse_remaining_facets",
    "compute_boundary",
    "compute_cone",
    "compute_elementary_collapse",
    "compute_f_vector",
    "compute_face_enumerator",
    "compute_g_vector",
    "compute_induced_subcomplex",
    "compute_join",
    "compute_link",
    "compute_skeleton",
    "compute_star",
    "compute_vertex_deletion",
    "join_maximal_facets",
    "skeleton_maximal_facets",
]
