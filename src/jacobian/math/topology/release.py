"""Small, composable finite-complex transforms for the topology release."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_FACETS,
    FiniteSimplicialComplex,
    HomologyConvention,
    Simplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology._structural import _maximal_faces
from jacobian.math.topology.operations import canonicalize, homology


class FacePosetRequest(StrictModel):
    complex: SimplicialComplexRequest


class FacePosetResult(StrictModel):
    complex: FiniteSimplicialComplex
    faces: tuple[Simplex, ...]
    order_relations: tuple[tuple[int, int], ...]
    order_complex: FiniteSimplicialComplex


class CliqueRequest(StrictModel):
    complex: SimplicialComplexRequest


class CliqueResult(StrictModel):
    source: FiniteSimplicialComplex
    graph_edges: tuple[tuple[str, str], ...]
    clique_facets: tuple[Simplex, ...]
    clique_complex: FiniteSimplicialComplex


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


# These are operation-envelope bounds, separate from the canonical complex
# carrier.  In particular, a barycentric order complex can have many more
# facets than its source has faces.
MAX_RELEASE_RELATIONS = 131_072
MAX_RELEASE_CLIQUE_SEARCH = 1_000_000


def _canonical(request: SimplicialComplexRequest) -> FiniteSimplicialComplex:
    return canonicalize(request.vertices, request.facets).complex


def _all_faces_sorted(complex_: FiniteSimplicialComplex) -> tuple[Simplex, ...]:
    return tuple(face for group in complex_.faces_by_dimension for face in group.faces)


def face_poset(request: FacePosetRequest) -> FacePosetResult:
    complex_ = _canonical(request.complex)
    faces = _all_faces_sorted(complex_)
    face_sets = tuple(set(face) for face in faces)
    relation_count = sum(
        1
        for i, a in enumerate(face_sets)
        for j, b in enumerate(face_sets)
        if len(a) < len(b) and a.issubset(b)
    )
    if relation_count > MAX_RELEASE_RELATIONS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.face_poset.relation_budget",
            message="the face-poset relation set exceeds the admitted bound",
        )
    relations = tuple(
        (i, j)
        for i, a in enumerate(face_sets)
        for j, b in enumerate(face_sets)
        if len(a) < len(b) and a.issubset(b)
    )
    successors: list[list[int]] = [[] for _ in faces]
    predecessors: list[list[int]] = [[] for _ in faces]
    for i, j in relations:
        if len(faces[j]) == len(faces[i]) + 1:
            successors[i].append(j)
            predecessors[j].append(i)
    # Maximal chains are generated directly in the ranked face poset.  The
    # former combinations() call considered every subset of the face set.
    chains: list[tuple[str, ...]] = []
    chain_count = 0
    for start in range(len(faces)):
        if predecessors[start]:
            continue
        stack: list[tuple[int, ...]] = [(start,)]
        while stack:
            chain = stack.pop()
            next_nodes = [j for j in successors[chain[-1]] if j not in chain]
            if not next_nodes:
                chain_count += 1
                if chain_count > MAX_TOPOLOGY_FACETS:
                    raise OperationResourceAdmissionError(
                        location=("complex",),
                        code="topology.face_poset.output_budget",
                        message="the order complex has too many maximal chains",
                    )
                chains.append(tuple(f"f{index}" for index in chain))
            else:
                stack.extend((*chain, j) for j in next_nodes)
    facets = tuple(chains)
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
    neighbors: dict[str, set[str]] = {
        vertex: set() for vertex in source.vertices
    }
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    facets: list[Simplex] = []
    explored = 0

    def bron_kerbosch(r: set[str], p: set[str], x: set[str]) -> None:
        nonlocal explored
        explored += 1
        if explored > MAX_RELEASE_CLIQUE_SEARCH:
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="topology.clique_complex.search_budget",
                message="maximal-clique search exceeds the admitted work bound",
            )
        if not p and not x:
            if len(facets) >= MAX_TOPOLOGY_FACETS:
                raise OperationResourceAdmissionError(
                    location=("complex",),
                    code="topology.clique_complex.output_budget",
                    message="the clique complex has too many maximal cliques",
                )
            facets.append(tuple(sorted(r)))
            return
        pivot = max(p | x, key=lambda v: len(neighbors[v]), default=None)
        candidates = p - (neighbors[pivot] if pivot is not None else set())
        for vertex in tuple(sorted(candidates)):
            bron_kerbosch(r | {vertex}, p & neighbors[vertex], x & neighbors[vertex])
            p.remove(vertex)
            x.add(vertex)

    bron_kerbosch(set(), set(source.vertices), set())
    maximal = tuple(sorted(facets))
    result_complex = canonicalize(source.vertices, maximal).complex
    return CliqueResult(
        source=source,
        graph_edges=edges,
        clique_facets=maximal,
        clique_complex=result_complex,
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
    # A 0-manifold has singleton facets and no codimension-one adjacency.
    # The empty ridge is not a shared geometric face in this dimension.
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
    complex_ = _canonical(request.complex)
    checked = _all_faces_sorted(complex_)
    # For a d-manifold, reduced link homology is that of S^(d-|sigma|-1).
    for face in checked:
        local = local_homology(
            LocalHomologyRequest(
                complex=request.complex, simplex=face, prime=request.prime
            )
        )
        expected_dim = complex_.dimension - len(face)
        expected = tuple(
            1 if i == expected_dim else 0 for i in range(max(expected_dim + 1, 1))
        )
        actual = local.betti_numbers
        if expected_dim < 0:
            continue
        if tuple(actual[: len(expected)]) != expected or any(actual[len(expected) :]):
            return HomologyManifoldResult(
                complex=complex_,
                prime=request.prime,
                homology_manifold=False,
                checked_faces=checked,
                obstruction_face=face,
            )
    return HomologyManifoldResult(
        complex=complex_,
        prime=request.prime,
        homology_manifold=True,
        checked_faces=checked,
    )


__all__ = [
    "CliqueRequest",
    "CliqueResult",
    "FacePosetRequest",
    "FacePosetResult",
    "HomologyManifoldRequest",
    "HomologyManifoldResult",
    "LocalHomologyRequest",
    "LocalHomologyResult",
    "OrientabilityRequest",
    "OrientabilityResult",
    "clique_complex",
    "face_poset",
    "homology_manifold",
    "local_homology",
    "orientability",
]
