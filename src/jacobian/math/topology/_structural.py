"""Local simplicial-transform contracts and deterministic kernels.

This module owns the contracts whose postconditions are local transforms of a
finite simplicial complex.  The canonical complex and chain values deliberately
remain in :mod:`_models`; homology, shelling, barycentric subdivision, and
pseudomanifolds retain their already separate owners.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FiniteSimplicialComplex,
    Simplex,
    SimplicialComplexRequest,
    VertexLabel,
    _validation_error,
    canonical_complex,
)
from jacobian.math.topology._request_admission import require_complex_admission


def _all_nonempty_faces(facets: tuple[Simplex, ...]) -> set[Simplex]:
    faces: set[Simplex] = set()
    for facet in facets:
        canonical = tuple(sorted(facet))
        for size in range(1, len(canonical) + 1):
            faces.update(combinations(canonical, size))
    return faces


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
        if complex_value is not None:
            raise _validation_error(
                "topology.require_complex_matches_facets_1", empty_message
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
    if all(frozenset(facet).issubset(deleted) for facet in complex_.facets):
        raise ValueError("deletion must leave at least one simplex")


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


class FVectorResult(StrictModel):
    """The f-vector and h-vector of a simplicial complex."""

    f_vector: tuple[int, ...]
    h_vector: tuple[int, ...]
    euler_characteristic: int
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

    The deletion must leave at least one simplex on the remaining vertices;
    deleting every vertex is rejected because the empty complex has no
    canonical value.
    """

    complex: SimplicialComplexRequest
    vertices_to_delete: tuple[VertexLabel, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_VERTICES,
        description="Vertex subset to remove. The deletion must leave at least one simplex on the remaining vertices: deleting every vertex is rejected because the empty complex has no canonical value.",
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
    skeleton_complex: FiniteSimplicialComplex | None = None

    @model_validator(mode="after")
    def require_structural_skeleton(self) -> Self:
        if not self.skeleton_facets:
            if self.skeleton_complex is not None:
                raise _validation_error(
                    "topology.require_skeleton_canonical_1",
                    "empty skeleton must have no complex",
                )
        else:
            if self.skeleton_complex is None:
                raise _validation_error(
                    "topology.require_skeleton_canonical_2",
                    "non-empty skeleton requires skeleton_complex",
                )
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
    width = max(map(len, complex_a.facets)) + max(map(len, complex_b.facets))
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
    join_complex: FiniteSimplicialComplex | None = None

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
    dimension = max(len(simplex) - 1 for simplex in all_simplices)
    f_vector = tuple(
        sum(len(simplex) == degree + 1 for simplex in all_simplices)
        for degree in range(dimension + 1)
    )
    from math import comb

    ring_dimension = dimension + 1
    h_vector = tuple(
        sum(
            (-1) ** (degree - index)
            * comb(ring_dimension - index, degree - index)
            * (1 if index == 0 else f_vector[index - 1])
            for index in range(degree + 1)
        )
        for degree in range(ring_dimension + 1)
    )
    return FVectorResult(
        f_vector=f_vector,
        h_vector=h_vector,
        euler_characteristic=sum(
            (-1) ** degree * count for degree, count in enumerate(f_vector)
        ),
        dimension=dimension,
    )


def compute_link(request: LinkRequest) -> LinkResult:
    require_complex_admission(request.complex)
    _require_simplex_in_complex(request.complex, request.simplex)
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
    _require_simplex_in_complex(request.complex, request.simplex)
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
    _require_deletion(request.complex, request.vertices_to_delete)
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
        skeleton_complex=canonical_complex(vertices, facets) if facets else None,
    )


def compute_join(request: JoinRequest) -> JoinResult:
    require_complex_admission(request.complex_a)
    require_complex_admission(request.complex_b)
    _require_join_admission(request.complex_a, request.complex_b)
    facets = join_maximal_facets(request.complex_a.facets, request.complex_b.facets)
    vertices = tuple(
        sorted(set(request.complex_a.vertices) | set(request.complex_b.vertices))
    )
    dimension = max((len(facet) - 1 for facet in facets), default=0)
    return JoinResult._from_kernel(
        complex_a=request.complex_a,
        complex_b=request.complex_b,
        join_vertices=vertices,
        join_facets=facets,
        join_dimension=dimension,
        join_complex=canonical_complex(vertices, facets) if facets else None,
    )


def compute_elementary_collapse(
    request: ElementaryCollapseRequest,
) -> ElementaryCollapseResult:
    require_complex_admission(request.complex)
    _require_collapse(request.complex, request.free_face, request.coface)
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
    "ElementaryCollapseRequest",
    "ElementaryCollapseResult",
    "FVectorRequest",
    "FVectorResult",
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
    "compute_elementary_collapse",
    "compute_f_vector",
    "compute_join",
    "compute_link",
    "compute_skeleton",
    "compute_star",
    "compute_vertex_deletion",
    "join_maximal_facets",
    "skeleton_maximal_facets",
]
