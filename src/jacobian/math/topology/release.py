"""Small, composable finite-complex transforms for the topology release."""

from __future__ import annotations

from itertools import combinations
from math import comb

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_VERTICES,
    FiniteSimplicialComplex,
    HomologyConvention,
    Simplex,
    SimplicialComplexRequest,
    is_bounded_prime,
)
from jacobian.math.topology._structural import _maximal_faces
from jacobian.math.topology.operations import canonicalize, homology

MAX_FACE_POSET_RELATIONS = 16_384
MAX_FACE_POSET_CHAINS = 16_384
MAX_FACE_POSET_PAIR_CANDIDATES = 1_000_000
MAX_FACE_POSET_CHAIN_CANDIDATES = 100_000
MAX_CLIQUE_CANDIDATES = 100_000
MAX_CLIQUE_FACETS = 16_384
MAX_GRAPH_CLIQUE_VERTICES = 8


class FacePosetRequest(StrictModel):
    complex: SimplicialComplexRequest


class FacePosetResult(StrictModel):
    complex: FiniteSimplicialComplex
    faces: tuple[Simplex, ...]
    order_relations: tuple[tuple[int, int], ...]
    order_complex: FiniteSimplicialComplex


class CliqueRequest(StrictModel):
    complex: SimplicialComplexRequest


class GraphCliqueRequest(StrictModel):
    graph: IndexedSimpleUndirectedGraph


class CliqueResult(StrictModel):
    source: FiniteSimplicialComplex
    graph_edges: tuple[tuple[str, str], ...]
    clique_facets: tuple[Simplex, ...]
    clique_complex: FiniteSimplicialComplex


class OneSkeletonRequest(StrictModel):
    complex: SimplicialComplexRequest


class OneSkeletonResult(StrictModel):
    """A graph value with its exact simplicial vertex and edge axes."""

    source: FiniteSimplicialComplex
    graph: IndexedSimpleUndirectedGraph
    vertex_labels: tuple[str, ...] = Field(max_length=MAX_TOPOLOGY_VERTICES)
    edge_faces: tuple[Simplex, ...] = Field(
        max_length=MAX_TOPOLOGY_VERTICES * (MAX_TOPOLOGY_VERTICES - 1) // 2
    )

    @model_validator(mode="after")
    def require_source_axes(self) -> OneSkeletonResult:
        if self.vertex_labels != self.source.vertices:
            raise ValueError("vertex_labels must equal the source vertex axis")
        if self.graph.vertex_count != len(self.source.vertices):
            raise ValueError("graph vertex_count must match the source vertex axis")
        graph_faces = tuple(
            tuple(self.vertex_labels[index] for index in edge)
            for edge in self.graph.edges
        )
        if graph_faces != self.edge_faces:
            raise ValueError("edge_faces must map graph edges in graph edge order")
        source_edges = (
            self.source.faces_by_dimension[1].faces
            if self.source.dimension >= 1
            else ()
        )
        if set(self.edge_faces) != set(source_edges):
            raise ValueError("graph edges must correspond exactly to source 1-faces")
        return self


class OrientabilityRequest(StrictModel):
    complex: SimplicialComplexRequest


class OrientabilityResult(StrictModel):
    complex: FiniteSimplicialComplex
    orientable: bool
    facet_signs: tuple[int, ...]
    obstruction_ridge: Simplex | None = None
    pure: bool = True


class LocalHomologyRequest(StrictModel):
    complex: SimplicialComplexRequest
    simplex: tuple[str, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=251, default=2)


class LocalHomologyResult(StrictModel):
    complex: FiniteSimplicialComplex
    simplex: Simplex
    link: FiniteSimplicialComplex | None
    reduced: bool
    prime: int
    betti_numbers: tuple[int, ...]
    empty_link: bool


class HomologyManifoldRequest(StrictModel):
    complex: SimplicialComplexRequest
    prime: int = Field(ge=2, le=251, default=2)


class HomologyManifoldResult(StrictModel):
    complex: FiniteSimplicialComplex
    prime: int
    homology_manifold: bool
    checked_faces: tuple[Simplex, ...]
    obstruction_face: Simplex | None = None


def _canonical(request: SimplicialComplexRequest) -> FiniteSimplicialComplex:
    return canonicalize(request.vertices, request.facets).complex


def one_skeleton(request: OneSkeletonRequest) -> OneSkeletonResult:
    """Return the graph on the canonical vertex axis and map edges to faces."""
    source = _canonical(request.complex)
    faces = source.faces_by_dimension[1].faces if source.dimension >= 1 else ()
    vertex_index = {label: index for index, label in enumerate(source.vertices)}
    edges = tuple(
        sorted((vertex_index[left], vertex_index[right]) for left, right in faces)
    )
    graph = IndexedSimpleUndirectedGraph(vertex_count=len(source.vertices), edges=edges)
    edge_faces = tuple(
        tuple(source.vertices[index] for index in edge) for edge in graph.edges
    )
    return OneSkeletonResult(
        source=source,
        graph=graph,
        vertex_labels=source.vertices,
        edge_faces=edge_faces,
    )


def _all_faces_sorted(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


def face_poset(request: FacePosetRequest) -> FacePosetResult:
    complex_ = _canonical(request.complex)
    faces = _all_faces_sorted(complex_)
    if len(faces) * len(faces) > MAX_FACE_POSET_PAIR_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.face_poset.pair_budget",
            message="face-poset relation candidates exceed the admitted work bound",
        )
    chain_candidates = sum(
        comb(len(faces), size) for size in range(1, complex_.dimension + 2)
    )
    if chain_candidates > MAX_FACE_POSET_CHAIN_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.face_poset.chain_candidate_budget",
            message="order-complex chain candidates exceed the admitted work bound",
        )
    relations_list: list[tuple[int, int]] = []
    for i, a in enumerate(faces):
        for j, b in enumerate(faces):
            if len(a) < len(b) and set(a).issubset(b):
                if len(relations_list) >= MAX_FACE_POSET_RELATIONS:
                    raise OperationResourceAdmissionError(
                        location=("complex",),
                        code="topology.face_poset.relation_budget",
                        message="face-poset relations exceed the admitted output bound",
                    )
                relations_list.append((i, j))
    relations = tuple(relations_list)
    relation_set = set(relations)
    chains: list[tuple[str, ...]] = []
    for size in range(1, complex_.dimension + 2):
        for chain in combinations(range(len(faces)), size):
            if all((chain[i], chain[i + 1]) in relation_set for i in range(size - 1)):
                if len(chains) >= MAX_FACE_POSET_CHAINS:
                    raise OperationResourceAdmissionError(
                        location=("complex",),
                        code="topology.face_poset.chain_budget",
                        message="order-complex simplices exceed the admitted output bound",
                    )
                chains.append(tuple(f"f{index}" for index in chain))
    facets = _maximal_faces(chains)
    order_complex = canonicalize(
        tuple(f"f{i}" for i in range(len(faces))), facets
    ).complex
    return FacePosetResult(
        complex=complex_,
        faces=faces,
        order_relations=relations,
        order_complex=order_complex,
    )


def clique_complex(request: CliqueRequest) -> CliqueResult:
    source = _canonical(request.complex)
    edges: tuple[tuple[str, str], ...] = (
        tuple((face[0], face[1]) for face in source.faces_by_dimension[1].faces)
        if source.dimension >= 1
        else ()
    )
    edge_set = {frozenset(edge) for edge in edges}
    facets: list[Simplex] = []
    vertices = source.vertices
    # A flag complex is determined by its graph, not by the dimension of the
    # presentation supplied by the caller.  In particular, a K4 presented as
    # a one-dimensional graph still has a 3-simplex.  Admit every possible
    # clique size before beginning enumeration so the search bound covers the
    # complete candidate expansion.
    candidate_count = 0
    for size in range(1, len(vertices) + 1):
        candidate_count += comb(len(vertices), size)
        if candidate_count > MAX_CLIQUE_CANDIDATES:
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="topology.clique.candidate_budget",
                message="clique candidates exceed the admitted search bound",
            )
    for size in range(1, len(vertices) + 1):
        for candidate in combinations(vertices, size):
            if size == 1 or all(
                frozenset(pair) in edge_set for pair in combinations(candidate, 2)
            ):
                facets.append(candidate)
    maximal = _maximal_faces(facets)
    if len(maximal) > MAX_CLIQUE_FACETS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.clique.output_budget",
            message="clique facets exceed the admitted output bound",
        )
    result_complex = canonicalize(vertices, maximal).complex
    return CliqueResult(
        source=source,
        graph_edges=edges,
        clique_facets=maximal,
        clique_complex=result_complex,
    )


def graph_clique_complex(request: GraphCliqueRequest) -> CliqueResult:
    """Return the flag complex of a bounded indexed graph.

    The graph is encoded as a one-dimensional finite simplicial complex and
    passed through the same exact clique kernel used by complex completion.
    Eight vertices is the largest envelope whose entire nonempty powerset
    stays within the canonical simplicial carrier's dimension-seven limit.
    """
    graph = request.graph
    if not 1 <= graph.vertex_count <= MAX_GRAPH_CLIQUE_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph", "vertex_count"),
            code="topology.graph_clique.vertex_budget",
            message="graph clique complexes admit between 1 and 8 vertices",
        )
    vertices = tuple(f"v{index}" for index in range(graph.vertex_count))
    endpoints = {vertex for edge in graph.edges for vertex in edge}
    facets = tuple((vertices[left], vertices[right]) for left, right in graph.edges)
    facets += tuple(
        (vertices[index],)
        for index in range(graph.vertex_count)
        if index not in endpoints
    )
    return clique_complex(
        CliqueRequest(
            complex=SimplicialComplexRequest(vertices=vertices, facets=facets)
        )
    )


def orientability(request: OrientabilityRequest) -> OrientabilityResult:
    complex_ = _canonical(request.complex)
    facets = complex_.maximal_simplices
    dimension = complex_.dimension
    if any(len(facet) != dimension + 1 for facet in facets):
        return OrientabilityResult(
            complex=complex_,
            orientable=False,
            pure=False,
            facet_signs=tuple(0 for _ in facets),
        )
    # In dimension zero the codimension-one face is the empty simplex, which
    # is not an adjacency ridge: distinct vertices are independent components.
    # Every finite 0-manifold is orientable, with an independently chosen sign.
    if dimension == 0:
        return OrientabilityResult(
            complex=complex_,
            orientable=True,
            pure=True,
            facet_signs=tuple(1 for _ in facets),
        )
    adjacency: dict[int, list[tuple[int, Simplex, int, int]]] = {
        i: [] for i in range(len(facets))
    }
    ridge_owner: dict[frozenset[str], list[tuple[int, int]]] = {}
    for i, facet in enumerate(facets):
        for pos in range(dimension + 1):
            ridge = tuple(v for j, v in enumerate(facet) if j != pos)
            ridge_owner.setdefault(frozenset(ridge), []).append((i, pos))
    for ridge_key, owners in ridge_owner.items():
        if len(owners) > 2:
            return OrientabilityResult(
                complex=complex_,
                orientable=False,
                pure=True,
                facet_signs=tuple(0 for _ in facets),
                obstruction_ridge=tuple(sorted(ridge_key)),
            )
        if len(owners) == 2:
            (a, pa), (b, pb) = owners
            adjacency[a].append((b, tuple(sorted(ridge_key)), pa, pb))
            adjacency[b].append((a, tuple(sorted(ridge_key)), pb, pa))
    signs: dict[int, int] = {}
    for root in range(len(facets)):
        if root in signs:
            continue
        signs[root] = 1
        stack = [root]
        while stack:
            a = stack.pop()
            for b, ridge, pa, pb in adjacency[a]:
                # induced ridge signs must be opposite
                wanted = -signs[a] * ((-1) ** (pa + pb))
                if b in signs and signs[b] != wanted:
                    return OrientabilityResult(
                        complex=complex_,
                        orientable=False,
                        pure=True,
                        facet_signs=tuple(signs.get(i, 0) for i in range(len(facets))),
                        obstruction_ridge=ridge,
                    )
                if b not in signs:
                    signs[b] = wanted
                    stack.append(b)
    return OrientabilityResult(
        complex=complex_,
        orientable=True,
        pure=True,
        facet_signs=tuple(signs[i] for i in range(len(facets))),
    )


def local_homology(request: LocalHomologyRequest) -> LocalHomologyResult:
    if not is_bounded_prime(request.prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="topology.local_homology.prime_not_prime",
            message="local homology coefficients require a prime field modulus",
        )
    complex_ = _canonical(request.complex)
    simplex = tuple(sorted(request.simplex))
    faces = set(_all_faces_sorted(complex_))
    if simplex not in faces:
        raise OperationDomainValidationError(
            location=("simplex",),
            code="topology.local_homology.not_a_face",
            message="simplex must be a face of the complex",
        )
    target = set(simplex)
    link_facets = _maximal_faces(
        tuple(
            tuple(sorted(set(f) - target))
            for f in complex_.maximal_simplices
            if target.issubset(f) and set(f) - target
        )
    )
    if not link_facets:
        return LocalHomologyResult(
            complex=complex_,
            simplex=simplex,
            link=None,
            reduced=True,
            prime=request.prime,
            betti_numbers=(1,),
            empty_link=True,
        )
    link_vertices = tuple(sorted({v for f in link_facets for v in f}))
    link = canonicalize(link_vertices, link_facets).complex
    result = homology(link, request.prime, HomologyConvention.REDUCED)
    return LocalHomologyResult(
        complex=complex_,
        simplex=simplex,
        link=link,
        reduced=True,
        prime=request.prime,
        betti_numbers=tuple(group.betti_number for group in result.groups),
        empty_link=False,
    )


def homology_manifold(request: HomologyManifoldRequest) -> HomologyManifoldResult:
    if not is_bounded_prime(request.prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="topology.homology_manifold.prime_not_prime",
            message="homology-manifold coefficients require a prime field modulus",
        )
    complex_ = _canonical(request.complex)
    all_faces = _all_faces_sorted(complex_)
    # A manifold carrier is pure before any local homology is interpreted.
    # This rejects, for example, a circle together with an isolated vertex;
    # its empty link is not a sphere of the ambient dimension.
    if any(
        len(facet) != complex_.dimension + 1 for facet in complex_.maximal_simplices
    ):
        return HomologyManifoldResult(
            complex=complex_,
            prime=request.prime,
            homology_manifold=False,
            checked_faces=(),
            obstruction_face=next(
                facet
                for facet in complex_.maximal_simplices
                if len(facet) != complex_.dimension + 1
            ),
        )
    checked: list[Simplex] = []
    # For a d-manifold, reduced link homology is that of S^(d-|sigma|-1).
    # The empty link is the degree -1 reduced group H~_-1(empty)=field,
    # represented by local_homology's explicit empty_link flag and (1,).
    for face in all_faces:
        local = local_homology(
            LocalHomologyRequest(
                complex=request.complex, simplex=face, prime=request.prime
            )
        )
        expected_dim = complex_.dimension - len(face)
        checked.append(face)
        if expected_dim == -1:
            if not local.empty_link or local.betti_numbers != (1,):
                return HomologyManifoldResult(
                    complex=complex_,
                    prime=request.prime,
                    homology_manifold=False,
                    checked_faces=tuple(checked),
                    obstruction_face=face,
                )
            continue
        # Non-maximal faces must have a nonempty link; a missing link is not
        # silently identified with the degree -1 group.
        if local.empty_link:
            return HomologyManifoldResult(
                complex=complex_,
                prime=request.prime,
                homology_manifold=False,
                checked_faces=tuple(checked),
                obstruction_face=face,
            )
        expected = tuple(1 if i == expected_dim else 0 for i in range(expected_dim + 1))
        actual = local.betti_numbers
        if tuple(actual[: len(expected)]) != expected or any(actual[len(expected) :]):
            return HomologyManifoldResult(
                complex=complex_,
                prime=request.prime,
                homology_manifold=False,
                checked_faces=tuple(checked),
                obstruction_face=face,
            )
    return HomologyManifoldResult(
        complex=complex_,
        prime=request.prime,
        homology_manifold=True,
        checked_faces=tuple(checked),
    )


__all__ = [
    "CliqueRequest",
    "CliqueResult",
    "FacePosetRequest",
    "FacePosetResult",
    "GraphCliqueRequest",
    "HomologyManifoldRequest",
    "HomologyManifoldResult",
    "LocalHomologyRequest",
    "LocalHomologyResult",
    "OneSkeletonRequest",
    "OneSkeletonResult",
    "OrientabilityRequest",
    "OrientabilityResult",
    "clique_complex",
    "face_poset",
    "graph_clique_complex",
    "homology_manifold",
    "local_homology",
    "one_skeleton",
    "orientability",
]
