"""Small, composable finite-complex transforms for the topology release."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from pydantic import Field, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    ElementLabel,
    FinitePoset,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
    verify_finite_poset,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_FACES,
    MAX_TOPOLOGY_FACETS,
    MAX_TOPOLOGY_VERTICES,
    FiniteSimplicialComplex,
    HomologyConvention,
    Simplex,
    SimplicialComplexRequest,
    canonical_complex,
    is_bounded_prime,
)
from jacobian.math.topology._structural import _maximal_faces
from jacobian.math.topology.operations import canonicalize, homology

MAX_FACE_POSET_RELATIONS = 16_384
MAX_FACE_POSET_CHAINS = 16_384
MAX_FACE_POSET_PAIR_CANDIDATES = 1_000_000
MAX_FACE_POSET_CHAIN_CANDIDATES = 100_000
MAX_CLIQUE_CANDIDATES = 100_000
MAX_CLIQUE_PAIR_CHECKS = 2_000_000
MAX_ORDER_COMPLEX_WORK = 1_000_000
MAX_ORDER_COMPLEX_OUTPUT_BYTES = 1_500_000


@dataclass(frozen=True)
class _OrderComplexPlan:
    strict_successors: dict[str, list[str]]
    cover_predecessors: dict[str, list[str]]
    cover_successors: dict[str, list[str]]
    max_chain_length: int


def _order_complex_plan(poset: FinitePoset) -> _OrderComplexPlan:
    elements = poset.elements
    strict_predecessors: dict[str, list[str]] = {element: [] for element in elements}
    strict_successors: dict[str, list[str]] = {element: [] for element in elements}
    cover_predecessors: dict[str, list[str]] = {element: [] for element in elements}
    cover_successors: dict[str, list[str]] = {element: [] for element in elements}
    for pair in poset.strict_order_pairs:
        strict_predecessors[pair.upper].append(pair.lower)
        strict_successors[pair.lower].append(pair.upper)
    for pair in poset.cover_relations:
        cover_predecessors[pair.upper].append(pair.lower)
        cover_successors[pair.lower].append(pair.upper)

    topological_order: list[str] = []
    remaining = set(elements)
    while remaining:
        ready = sorted(
            element
            for element in remaining
            if not set(cover_predecessors[element]).intersection(remaining)
        )
        if not ready:  # verify_finite_poset has already established acyclicity.
            raise OperationDomainValidationError(
                location=("poset",),
                code="topology.order_complex.invalid_poset",
                message="poset cover relations are not acyclic",
            )
        topological_order.extend(ready)
        remaining.difference_update(ready)

    chains_ending: dict[str, int] = {}
    incidences_ending: dict[str, int] = {}
    longest_chain_ending: dict[str, int] = {}
    maximal_paths_ending: dict[str, int] = {}
    for element in topological_order:
        chains_ending[element] = 1 + sum(
            chains_ending[lower] for lower in strict_predecessors[element]
        )
        incidences_ending[element] = 1 + sum(
            incidences_ending[lower] + chains_ending[lower]
            for lower in strict_predecessors[element]
        )
        longest_chain_ending[element] = 1 + max(
            (longest_chain_ending[lower] for lower in cover_predecessors[element]),
            default=0,
        )
        maximal_paths_ending[element] = (
            1
            if not cover_predecessors[element]
            else sum(
                maximal_paths_ending[lower] for lower in cover_predecessors[element]
            )
        )

    chain_count = sum(chains_ending.values())
    chain_vertex_incidences = sum(incidences_ending.values())
    max_chain_length = max(longest_chain_ending.values(), default=0)
    facet_count = sum(
        maximal_paths_ending[element]
        for element in elements
        if not cover_successors[element]
    )
    work_estimate = (
        len(elements) ** 2
        + len(elements) ** 3
        + len(poset.strict_order_pairs)
        + len(poset.cover_relations)
        + chain_count
        + chain_vertex_incidences
        + facet_count * max_chain_length
        + facet_count**2 * max_chain_length
    )
    output_bytes_estimate = (
        4096
        + 40 * len(elements)
        + 112
        * (
            len(poset.strict_order_pairs)
            + len(poset.cover_relations)
            + len(poset.incomparable_pairs)
        )
        + 40 * chain_vertex_incidences
        + 40 * facet_count * max_chain_length
    )
    _admit_order_complex_output(
        chain_count,
        max_chain_length,
        facet_count,
        work_estimate,
        output_bytes_estimate,
    )
    return _OrderComplexPlan(
        strict_successors=strict_successors,
        cover_predecessors=cover_predecessors,
        cover_successors=cover_successors,
        max_chain_length=max_chain_length,
    )


def _admit_order_complex_output(
    chain_count: int,
    max_chain_length: int,
    facet_count: int,
    work_estimate: int,
    output_bytes_estimate: int,
) -> None:
    if chain_count > MAX_TOPOLOGY_FACES:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="topology.order_complex.face_budget",
            message=(
                "the order complex has more nonempty chains than the finite "
                "simplicial-complex face bound"
            ),
        )
    if max_chain_length > MAX_TOPOLOGY_DIMENSION + 1:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="topology.order_complex.dimension_budget",
            message="the longest chain exceeds the finite-complex dimension bound",
        )
    if facet_count > MAX_TOPOLOGY_FACETS:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="topology.order_complex.facet_budget",
            message="maximal-chain facets exceed the finite-complex facet bound",
        )
    if work_estimate > MAX_ORDER_COMPLEX_WORK:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="topology.order_complex.work_budget",
            message="order-complex chain enumeration exceeds the admitted work bound",
        )
    if output_bytes_estimate > MAX_ORDER_COMPLEX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("poset",),
            code="topology.order_complex.output_budget",
            message="the complete order complex exceeds the admitted output bound",
        )


def _enumerate_order_complex_chains(
    elements: tuple[str, ...], plan: _OrderComplexPlan
) -> tuple[tuple[tuple[Simplex, ...], ...], tuple[tuple[str, ...], ...]]:
    chains_by_dimension: list[set[Simplex]] = [
        set() for _ in range(plan.max_chain_length)
    ]

    def collect_chains(prefix: tuple[str, ...], last: str) -> None:
        chains_by_dimension[len(prefix) - 1].add(tuple(sorted(prefix)))
        for successor in plan.strict_successors[last]:
            collect_chains((*prefix, successor), successor)

    for element in elements:
        collect_chains((element,), element)

    facet_chains: list[tuple[str, ...]] = []

    def collect_maximal_chains(prefix: tuple[str, ...], last: str) -> None:
        successors = plan.cover_successors[last]
        if not successors:
            facet_chains.append(prefix)
            return
        for successor in successors:
            collect_maximal_chains((*prefix, successor), successor)

    for element in elements:
        if not plan.cover_predecessors[element]:
            collect_maximal_chains((element,), element)

    ordered_facets = tuple(sorted(facet_chains, key=lambda chain: tuple(sorted(chain))))
    closure: tuple[tuple[Simplex, ...], ...] = tuple(
        tuple(sorted(faces)) for faces in chains_by_dimension if faces
    )
    return closure, ordered_facets


class FacePosetRequest(StrictModel):
    complex: SimplicialComplexRequest


class FacePosetResult(StrictModel):
    complex: FiniteSimplicialComplex
    faces: tuple[Simplex, ...]
    face_element_labels: tuple[ElementLabel, ...]
    order_relations: tuple[tuple[int, int], ...]
    poset: FinitePoset | None
    order_complex: FiniteSimplicialComplex


class OrderComplexRequest(StrictModel):
    poset: FinitePoset


class OrderComplexResult(StrictModel):
    poset: FinitePoset
    complex: FiniteSimplicialComplex
    vertex_elements: tuple[ElementLabel, ...]
    maximal_chains: tuple[tuple[ElementLabel, ...], ...]


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
        stored_source_edges = (
            self.source.faces_by_dimension[1].faces
            if self.source.dimension >= 1
            else ()
        )
        # The carrier's JSON decoder checks face-axis shape, but does not replay
        # facet closure. This operation relies specifically on the 1-face axis,
        # so check that bounded relation against at most 128 facets of size 8.
        facets = self.source.maximal_simplices
        if (
            not isinstance(facets, tuple)
            or len(facets) > MAX_TOPOLOGY_FACES
            or any(
                not isinstance(facet, tuple)
                or not 1 <= len(facet) <= MAX_TOPOLOGY_DIMENSION + 1
                for facet in facets
            )
        ):
            raise ValueError("source facets exceed the admitted shape bounds")
        if len(set(facets)) != len(facets):
            raise ValueError("source maximal facets must be unique")
        facet_edges = {
            tuple(sorted(pair)) for facet in facets for pair in combinations(facet, 2)
        }
        if set(stored_source_edges) != facet_edges:
            raise ValueError("source 1-face axis must match its maximal facets")
        if set(self.edge_faces) != facet_edges:
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
    if not isinstance(request, OneSkeletonRequest):
        raise OperationDomainValidationError(
            location=(),
            code="topology.one_skeleton.request_type",
            message="request must be a OneSkeletonRequest",
        )
    if not isinstance(request.complex, SimplicialComplexRequest):
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.one_skeleton.complex_type",
            message="complex must be a SimplicialComplexRequest",
        )
    try:
        complex_request = SimplicialComplexRequest.model_validate(
            request.complex.model_dump()
        )
    except (ValidationError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("complex",),
            code="topology.one_skeleton.invalid_complex",
            message="complex request fields are invalid",
        ) from error
    source = _canonical(complex_request)
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


def order_complex(request: OrderComplexRequest) -> OrderComplexResult:
    """Return the simplicial complex of all nonempty chains in a finite poset."""
    if not isinstance(request, OrderComplexRequest):
        raise OperationDomainValidationError(
            location=(),
            code="topology.order_complex.invalid_request",
            message="order-complex input must be an OrderComplexRequest",
        )
    poset = request.poset
    if not verify_finite_poset(poset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="topology.order_complex.invalid_poset",
            message="poset claims do not describe its canonical finite poset",
        )
    elements = poset.elements
    plan = _order_complex_plan(poset)
    closure, ordered_facets = _enumerate_order_complex_chains(elements, plan)
    complex_ = canonical_complex(elements, ordered_facets, closure=closure)
    return OrderComplexResult(
        poset=poset,
        complex=complex_,
        vertex_elements=elements,
        maximal_chains=ordered_facets,
    )


def face_poset(request: FacePosetRequest) -> FacePosetResult:
    if not isinstance(request, FacePosetRequest):
        raise OperationDomainValidationError(
            location=(),
            code="topology.face_poset.invalid_request",
            message="face-poset input must be a FacePosetRequest",
        )
    complex_ = _canonical(request.complex)
    faces = _all_faces_sorted(complex_)
    if len(faces) * len(faces) > MAX_FACE_POSET_PAIR_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.face_poset.pair_budget",
            message="face-poset relation candidates exceed the admitted work bound",
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
    face_element_labels = tuple(f"f{index:04d}" for index in range(len(faces)))
    if len(faces) <= 64:
        poset = materialize_finite_poset(
            face_element_labels,
            tuple(
                PresentationPair(
                    lower=face_element_labels[lower],
                    upper=face_element_labels[upper],
                )
                for lower, upper in relations
            ),
            RelationInterpretation.COMPARABLE_PAIRS,
            ReflexivePairPolicy.FORBIDDEN,
        )
        order_result = order_complex(OrderComplexRequest(poset=poset))
        return FacePosetResult(
            complex=complex_,
            faces=faces,
            face_element_labels=face_element_labels,
            order_relations=relations,
            poset=poset,
            order_complex=order_result.complex,
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
                chains.append(tuple(face_element_labels[index] for index in chain))
    facets = _maximal_faces(chains)
    order_complex_value = canonicalize(face_element_labels, facets).complex
    return FacePosetResult(
        complex=complex_,
        faces=faces,
        face_element_labels=face_element_labels,
        order_relations=relations,
        poset=None,
        order_complex=order_complex_value,
    )


def clique_complex(source: FiniteSimplicialComplex) -> CliqueResult:
    source = canonicalize(source.vertices, source.maximal_simplices).complex
    edges: tuple[tuple[str, str], ...] = (
        tuple((face[0], face[1]) for face in source.faces_by_dimension[1].faces)
        if source.dimension >= 1
        else ()
    )
    edge_set = {frozenset(edge) for edge in edges}
    vertices = source.vertices
    # A flag complex is determined by its graph, not by the dimension of the
    # presentation supplied by the caller.  In particular, a K4 presented as
    # a one-dimensional graph still has a 3-simplex. Preflight the whole
    # powerset and pair-check upper bound before the output-sensitive walk.
    candidate_count = 0
    pair_check_bound = 0
    for size in range(1, len(vertices) + 1):
        candidate_count += comb(len(vertices), size)
        if candidate_count > MAX_CLIQUE_CANDIDATES:
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="topology.clique.candidate_budget",
                message="clique candidates exceed the admitted search bound",
            )
        pair_check_bound += comb(len(vertices), size) * comb(size, 2)
        if pair_check_bound > MAX_CLIQUE_PAIR_CHECKS:
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="topology.clique.pair_check_budget",
                message="clique edge checks exceed the admitted work bound",
            )

    faces_by_dimension: list[list[Simplex]] = [
        [] for _ in range(MAX_TOPOLOGY_DIMENSION + 1)
    ]
    face_count = 0
    # A clique larger than the carrier's maximum simplex contains a
    # (MAX_TOPOLOGY_DIMENSION + 2)-vertex clique.  Checking that size is enough
    # to reject every out-of-carrier complex without enumerating larger sets.
    largest_candidate_size = min(len(vertices), MAX_TOPOLOGY_DIMENSION + 2)
    for size in range(largest_candidate_size, 0, -1):
        for candidate in combinations(vertices, size):
            if size > 1 and not all(
                frozenset(pair) in edge_set for pair in combinations(candidate, 2)
            ):
                continue
            if size > MAX_TOPOLOGY_DIMENSION + 1:
                raise OperationResourceAdmissionError(
                    location=("complex",),
                    code="topology.clique.dimension_budget",
                    message=(
                        "clique complex dimension exceeds the admitted "
                        f"maximum {MAX_TOPOLOGY_DIMENSION}"
                    ),
                )
            if face_count == MAX_TOPOLOGY_FACES:
                raise OperationResourceAdmissionError(
                    location=("complex",),
                    code="topology.clique.face_budget",
                    message=(
                        "clique complex face closure exceeds the admitted "
                        f"maximum {MAX_TOPOLOGY_FACES}"
                    ),
                )
            faces_by_dimension[size - 1].append(candidate)
            face_count += 1

    maximal: list[Simplex] = []
    maximal_sets: list[frozenset[str]] = []
    # Process larger faces first. Once a face is maximal, no later face can
    # contain it, so the public facet bound can be enforced before appending an
    # oversized maximal-facet collection.
    for faces in reversed(faces_by_dimension):
        for face in faces:
            face_set = frozenset(face)
            if any(existing.issuperset(face_set) for existing in maximal_sets):
                continue
            if len(maximal) == MAX_TOPOLOGY_FACETS:
                raise OperationResourceAdmissionError(
                    location=("complex",),
                    code="topology.clique.facet_budget",
                    message=(
                        "clique complex maximal facets exceed the admitted "
                        f"maximum {MAX_TOPOLOGY_FACETS}"
                    ),
                )
            maximal.append(face)
            maximal_sets.append(face_set)

    closure = tuple(tuple(faces) for faces in faces_by_dimension)
    if not vertices:
        result_complex = canonical_complex((), (), closure=())
        return CliqueResult(
            source=source,
            graph_edges=edges,
            clique_facets=(),
            clique_complex=result_complex,
        )
    highest_dimension = max(index for index, faces in enumerate(closure) if faces)
    closure = closure[: highest_dimension + 1]
    result_complex = canonical_complex(vertices, tuple(maximal), closure=closure)
    return CliqueResult(
        source=source,
        graph_edges=edges,
        clique_facets=tuple(maximal),
        clique_complex=result_complex,
    )


def graph_clique_complex(graph: IndexedSimpleUndirectedGraph) -> CliqueResult:
    """Return the flag complex of a bounded indexed graph.

    The graph is encoded as a one-dimensional finite simplicial complex and
    passed through the same exact clique kernel used by complex completion.
    Eight vertices is the largest envelope whose entire nonempty powerset
    stays within the canonical simplicial carrier's dimension-seven limit.
    """
    if not 1 <= graph.vertex_count <= MAX_TOPOLOGY_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph", "vertex_count"),
            code="topology.graph_clique.vertex_budget",
            message=(
                f"graph clique complexes admit between 1 and "
                f"{MAX_TOPOLOGY_VERTICES} vertices"
            ),
        )
    vertices = tuple(f"v{index}" for index in range(graph.vertex_count))
    endpoints = {vertex for edge in graph.edges for vertex in edge}
    facets: tuple[Simplex, ...] = tuple(
        (vertices[left], vertices[right]) for left, right in graph.edges
    )
    facets += tuple(
        (vertices[index],)
        for index in range(graph.vertex_count)
        if index not in endpoints
    )
    return clique_complex(canonicalize(vertices, facets).complex)


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
    "OrderComplexRequest",
    "OrderComplexResult",
    "OrientabilityRequest",
    "OrientabilityResult",
    "clique_complex",
    "face_poset",
    "graph_clique_complex",
    "homology_manifold",
    "local_homology",
    "one_skeleton",
    "order_complex",
    "orientability",
]
