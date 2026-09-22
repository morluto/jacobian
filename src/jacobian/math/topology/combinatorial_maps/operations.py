"""Exact native kernels over finite combinatorial maps.

All functions are deterministic and complete for accepted values.  They
need no ``UNKNOWN``, timeout-as-mathematics, search budget, or solver
outcome.  They use a small exact permutation/orbit kernel over immutable
dart IDs; no backend embedding object crosses the boundary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from itertools import pairwise, permutations
from math import factorial

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.multigraph._models import LooplessMultigraph
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices.values import (
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)

from ._face_orbits import face_orbit_data
from ._models import (
    MAX_EMBEDDING_DEGREE,
    MAX_EMBEDDING_EDGES,
    MAX_EMBEDDING_ROTATION_ENTRIES,
    MAX_EMBEDDING_VERTICES,
    MAX_MINIMUM_GENUS_CANDIDATES,
    MAX_MULTIGRAPH_EMBEDDING_EDGES,
    MAX_MULTIGRAPH_EMBEDDING_VERTICES,
    MAX_ROTATION_SYSTEM_CANDIDATES,
    CombinatorialMapBijection,
    ConnectedComponentsResult,
    DualResult,
    EulerCharacteristicCounts,
    EulerCharacteristicResult,
    FacesResult,
    MinimumGenusResult,
    MultigraphEmbeddingResult,
    OrientableEmbeddingCheckResult,
    OrientableGenusResult,
    OrientationReverseResult,
    RotationSystemFindResult,
    SignedEmbeddingCheckResult,
    VertexFaceIncidenceResult,
)
from ._signed_faces import signed_projected_faces
from .values import (
    FiniteCombinatorialMap,
    _build_outgoing,
    _validate_facial_budgets,
    _validate_involution,
    _validate_rotation,
)

__all__ = [
    "check_multigraph_embedding",
    "check_orientable_embedding",
    "check_signed_embedding",
    "connected_components",
    "connected_components_vertices",
    "dual_map",
    "euler_characteristic",
    "face_orbits",
    "find_rotation_system",
    "minimum_orientable_genus",
    "orientable_genus",
    "orientation_reverse",
    "rotation_successor",
    "verify_dual",
    "verify_minimum_orientable_genus",
    "verify_multigraph_embedding",
    "verify_orientable_embedding",
    "verify_orientation_reverse",
    "verify_signed_embedding",
    "verify_vertex_face_incidence",
    "vertex_face_incidence",
]


def _admit_map(
    map_: FiniteCombinatorialMap,
) -> tuple[list[list[int]], dict[int, int], list[int]]:
    """Admit map laws and compute bounded facial data once for the request."""
    try:
        _validate_involution(map_.darts)
        _validate_rotation(
            map_.rotations, _build_outgoing(map_.darts, map_.vertex_count), map_.darts
        )
        data = face_orbit_data(map_)
        _validate_facial_budgets(data[0])
        return data
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("map",), code=error.type, message=error.message()
        ) from error


def minimum_orientable_genus(
    graph: SimpleUndirectedGraph, max_candidates: int = MAX_MINIMUM_GENUS_CANDIDATES
) -> MinimumGenusResult:
    """Exhaustively compute the minimum orientable cellular genus of ``graph``.

    Unlike the existing threshold search, this operation only returns EXACT
    after the complete finite rotation-system family has been examined.  A
    budget stop is UNKNOWN and never a lower-bound or nonexistence claim.
    """
    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.minimum_genus.graph_carrier",
            message="graph must be a SimpleUndirectedGraph",
        )
    if (
        type(max_candidates) is not int
        or not 1 <= max_candidates <= MAX_MINIMUM_GENUS_CANDIDATES
    ):
        raise OperationDomainValidationError(
            location=("max_candidates",),
            code="topology.minimum_genus.candidate_bound",
            message="max_candidates is outside the admitted envelope",
        )
    _admit_embedding_candidate(graph, ())
    index, incident = _embedding_adjacency(graph)
    rows = _rotation_system_row_choices(incident)
    total_candidates = _rotation_system_total(incident)
    if not _embedding_connected(graph, index):
        return MinimumGenusResult._from_kernel(
            graph=graph,
            status="UNKNOWN",
            minimum_genus=None,
            rotations=(),
            certificate=None,
            candidates_examined=0,
            total_candidates=total_candidates,
            max_candidates=max_candidates,
            reason="GRAPH_DISCONNECTED",
        )
    best: OrientableEmbeddingCheckResult | None = None
    examined = 0
    for candidate in _lazy_product(rows):
        if examined >= max_candidates:
            return MinimumGenusResult._from_kernel(
                graph=graph,
                status="UNKNOWN",
                minimum_genus=None,
                rotations=(),
                certificate=None,
                candidates_examined=examined,
                total_candidates=total_candidates,
                max_candidates=max_candidates,
                reason="CANDIDATE_BUDGET_EXCEEDED",
            )
        checked = check_orientable_embedding(graph, candidate)
        examined += 1
        if checked.status != "ORIENTABLE_CELLULAR_EMBEDDING":
            continue
        if best is None or checked.genus < best.genus:
            best = checked
    if best is None:  # pragma: no cover - every connected unsigned system is cellular
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.minimum_genus.no_embedding",
            message="no cellular rotation system was produced",
        )
    return MinimumGenusResult._from_kernel(
        graph=graph,
        status="EXACT",
        minimum_genus=best.genus,
        rotations=best.rotations,
        certificate=best,
        candidates_examined=examined,
        total_candidates=total_candidates,
        max_candidates=max_candidates,
        reason=None,
    )


def verify_minimum_orientable_genus(claim: MinimumGenusResult) -> bool:
    """Check a caller-authored minimum-genus result against its retained graph."""
    if type(claim) is not MinimumGenusResult:
        return False
    try:
        expected = minimum_orientable_genus(claim.graph, claim.max_candidates)
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        return False
    return expected == claim


def check_multigraph_embedding(  # noqa: C901
    graph: LooplessMultigraph, rotations: tuple[tuple[str, ...], ...]
) -> MultigraphEmbeddingResult:
    """Validate the native rotation representation before any indexing."""
    if type(graph) is not LooplessMultigraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.multigraph_embedding.graph_carrier",
            message="graph must be a LooplessMultigraph",
        )
    if type(rotations) is not tuple or any(
        type(row) is not tuple or any(type(edge_id) is not str for edge_id in row)
        for row in rotations
    ):
        raise OperationDomainValidationError(
            location=("rotations",),
            code="topology.multigraph_embedding.rotations_shape",
            message="rotations must be a tuple of tuple rows containing string edge IDs",
        )
    if graph.vertex_count == 0:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.multigraph_embedding.zero_vertices",
            message="a multigraph embedding needs at least one vertex",
        )
    if (
        graph.vertex_count > MAX_MULTIGRAPH_EMBEDDING_VERTICES
        or len(graph.edges) > MAX_MULTIGRAPH_EMBEDDING_EDGES
    ):
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="topology.multigraph_embedding.size_bound",
            message="multigraph exceeds embedding envelope",
        )
    if len(rotations) != graph.vertex_count:
        return MultigraphEmbeddingResult._from_kernel(
            graph=graph,
            status="INVALID",
            rotations=rotations,
            obstruction="rotation row count does not match the source graph",
        )
    edge_by_id = {edge.edge_id: edge for edge in graph.edges}
    incidences: list[set[str]] = [set() for _ in range(graph.vertex_count)]
    for edge in graph.edges:
        incidences[edge.left].add(edge.edge_id)
        incidences[edge.right].add(edge.edge_id)
    for vertex, row in enumerate(rotations):
        if (
            len(row) != len(incidences[vertex])
            or set(row) != incidences[vertex]
            or len(set(row)) != len(row)
        ):
            return MultigraphEmbeddingResult._from_kernel(
                graph=graph,
                status="INVALID",
                rotations=rotations,
                obstruction=f"rotation row {vertex} does not cover its edge-ID incidences",
            )
    # Connectivity is part of the same closed-surface convention as the simple
    # graph checker.  A one-vertex edgeless multigraph is the sphere point.
    if graph.vertex_count > 1:
        reached = {0}
        changed = True
        while changed:
            changed = False
            for edge in graph.edges:
                if edge.left in reached or edge.right in reached:
                    before = len(reached)
                    reached.update((edge.left, edge.right))
                    changed |= len(reached) != before
        if len(reached) != graph.vertex_count:
            return MultigraphEmbeddingResult._from_kernel(
                graph=graph,
                status="INVALID",
                rotations=rotations,
                obstruction="source multigraph is disconnected",
            )
    darts: list[tuple[int, int, int]] = []
    edge_dart_ids: list[str] = []
    for i, edge in enumerate(graph.edges):
        darts.extend(
            ((edge.left, edge.right, 2 * i + 1), (edge.right, edge.left, 2 * i))
        )
        edge_dart_ids.extend((edge.edge_id, edge.edge_id))
    dart_rotations: list[tuple[int, ...]] = []
    for vertex, row in enumerate(rotations):
        dart_rotations.append(
            tuple(
                2 * list(edge_by_id).index(edge_id)
                + (0 if edge_by_id[edge_id].left == vertex else 1)
                for edge_id in row
            )
        )
    sigma: list[int] = [0] * len(darts)
    for dart_row in dart_rotations:
        for i, dart in enumerate(dart_row):
            sigma[dart] = dart_row[(i + 1) % len(dart_row)]
    alpha = [dart[2] for dart in darts]
    phi = [alpha[sigma[dart]] for dart in range(len(darts))] if darts else []
    walks: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(darts)):
        if start in seen:
            continue
        walk: list[int] = []
        current = start
        while current not in seen:
            seen.add(current)
            walk.append(current)
            current = phi[current]
        walks.append(tuple(walk))
    embedding_map = (
        FiniteCombinatorialMap(
            vertex_count=graph.vertex_count,
            darts=tuple(darts),
            rotations=tuple(dart_rotations),
        )
        if darts
        else None
    )
    face_of_dart: list[int] = [0] * len(darts)
    for face_index, face_walk in enumerate(walks):
        for dart in face_walk:
            face_of_dart[dart] = face_index
    dual = None
    if darts:
        dual_darts = tuple(
            (face_of_dart[dart], face_of_dart[reverse], reverse)
            for dart, (_tail, _head, reverse) in enumerate(darts)
        )
        dual = FiniteCombinatorialMap(
            vertex_count=len(walks), darts=dual_darts, rotations=tuple(walks)
        )
    return MultigraphEmbeddingResult._from_kernel(
        graph=graph,
        status="EMBEDDED",
        rotations=tuple(tuple(row) for row in rotations),
        darts=tuple(darts),
        edge_dart_ids=tuple(edge_dart_ids),
        face_walks=tuple(walks),
        dual_edge_ids=tuple(edge.edge_id for edge in graph.edges),
        embedding_map=embedding_map,
        dual_map=dual,
        obstruction=None,
    )


def verify_multigraph_embedding(claim: MultigraphEmbeddingResult) -> bool:
    if type(claim) is not MultigraphEmbeddingResult:
        return False
    return check_multigraph_embedding(claim.graph, claim.rotations) == claim


def rotation_successor(map_: FiniteCombinatorialMap, dart: int) -> int:
    """Return the dart following ``dart`` in its local rotation."""

    _admit_map(map_)
    if type(dart) is not int or not 0 <= dart < len(map_.darts):
        raise ValueError("dart must index the source map")
    tail = map_.darts[dart][0]
    row = map_.rotations[tail]
    index = row.index(dart)
    return row[(index + 1) % len(row)]


def _face_orbits(
    map_: FiniteCombinatorialMap,
) -> tuple[list[list[int]], dict[int, int], list[int], dict[int, list[int]]]:
    """Return the complete face-orbit family.

    The face permutation is ``face = reverse . rotation_successor`` applied to
    each dart: from a dart, advance to the next dart around its tail, then
    cross to the opposite dart to walk along the face on the other side.

    Returns ``(walks, face_of_dart, successor, per_component_faces)``:
    - ``walks``: a list of facial walks (each a list of dart indices)
    - ``face_of_dart``: ``dart -> face index``
    - ``successor``: the dart-successor permutation (``dart -> next dart``)
    - per-component face partition: ``component index -> face indices``
    """
    walks, face_of_dart, successor = _admit_map(map_)
    comp_of_vertex = _connected_components_vertices(map_)
    comp_of_face: dict[int, list[int]] = {}
    for face_index, walk in enumerate(walks):
        representative = walk[0]
        vertex = map_.darts[representative][0]
        comp = comp_of_vertex[vertex]
        comp_of_face.setdefault(comp, []).append(face_index)
    return walks, face_of_dart, successor, comp_of_face


def face_orbits(map_: FiniteCombinatorialMap) -> FacesResult:
    """Return the complete canonical face-orbit family."""

    walks, face_of_dart, successor, _ = _face_orbits(map_)
    return FacesResult.model_construct(
        map=map_,
        face_walks=tuple(tuple(walk) for walk in walks),
        face_of_dart=tuple(face_of_dart[dart] for dart in range(len(map_.darts))),
        successor=tuple(successor),
    )


def _connected_components_vertices(
    map_: FiniteCombinatorialMap,
) -> dict[int, int]:
    """Return ``vertex -> component index`` for the underlying graph."""

    parent = list(range(map_.vertex_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for dart in map_.darts:
        tail, head, _ = dart
        union(tail, head)
    comp_ids: dict[int, int] = {}
    result: dict[int, int] = {}
    for v in range(map_.vertex_count):
        root = find(v)
        if root not in comp_ids:
            comp_ids[root] = len(comp_ids)
        result[v] = comp_ids[root]
    return result


def connected_components_vertices(map_: FiniteCombinatorialMap) -> dict[int, int]:
    """Admit a map before returning its vertex component partition."""
    _admit_map(map_)
    return _connected_components_vertices(map_)


def connected_components(
    map_: FiniteCombinatorialMap,
) -> ConnectedComponentsResult:
    """Return the component partition of vertices, darts, and faces."""

    walks, _, _, _ = _face_orbits(map_)
    vertex_component = _connected_components_vertices(map_)
    face_component: dict[int, int] = {}
    for face_index, walk in enumerate(walks):
        representative = walk[0]
        vertex = map_.darts[representative][0]
        face_component[face_index] = vertex_component[vertex]
    dart_component: dict[int, int] = {}
    for dart_index, dart in enumerate(map_.darts):
        dart_component[dart_index] = vertex_component[dart[0]]
    return ConnectedComponentsResult.model_construct(
        vertex_component=tuple(
            vertex_component[vertex] for vertex in range(map_.vertex_count)
        ),
        dart_component=tuple(dart_component[dart] for dart in range(len(map_.darts))),
        face_component=tuple(face_component[face] for face in range(len(walks))),
    )


def euler_characteristic(
    map_: FiniteCombinatorialMap,
) -> EulerCharacteristicResult:
    """Return per-component and total Euler characteristic.

    Uses the disconnected-surface convention: each connected component is an
    independent closed surface, so ``chi = V - E + F`` per component and the
    total is the sum of component characteristics.
    """
    walks, _, _, _ = _face_orbits(map_)
    vertex_component = _connected_components_vertices(map_)
    component_vertices: dict[int, set[int]] = {}
    component_edges: dict[int, int] = {}
    for v in range(map_.vertex_count):
        comp = vertex_component[v]
        component_vertices.setdefault(comp, set()).add(v)
        component_edges.setdefault(comp, 0)
    for dart in map_.darts:
        comp = vertex_component[dart[0]]
        component_edges[comp] = component_edges.get(comp, 0) + 1
    for comp in component_edges:
        component_edges[comp] //= 2
    component_faces: dict[int, int] = {}
    for face_index in range(len(walks)):
        vertex = map_.darts[walks[face_index][0]][0]
        comp = vertex_component[vertex]
        component_faces[comp] = component_faces.get(comp, 0) + 1
    all_components = (
        set(component_vertices) | set(component_edges) | set(component_faces)
    )
    per_component: list[EulerCharacteristicCounts] = []
    total_v = total_e = total_f = 0
    for comp in sorted(all_components):
        v = len(component_vertices.get(comp, set()))
        e = component_edges.get(comp, 0)
        f = component_faces.get(comp, 0)
        per_component.append(
            EulerCharacteristicCounts(
                vertices=v,
                edges=e,
                faces=f,
                characteristic=v - e + f,
            )
        )
        total_v += v
        total_e += e
        total_f += f
    total = EulerCharacteristicCounts(
        vertices=total_v,
        edges=total_e,
        faces=total_f,
        characteristic=total_v - total_e + total_f,
    )
    return EulerCharacteristicResult.model_construct(
        per_component=tuple(per_component), total=total
    )


def orientable_genus(
    map_: FiniteCombinatorialMap,
) -> OrientableGenusResult:
    """Return per-component and total orientable genus.

    For each connected component, ``g = (2 - chi) / 2`` under the orientable
    cellular-map convention.  The result is an exact nonnegative integer for a
    valid orientable combinatorial map.  The total genus is the sum of the
    component genera.
    """
    profile = euler_characteristic(map_)
    component_genera: list[int] = []
    total = 0
    for row in profile.per_component:
        g = (2 - row.characteristic) // 2
        if (2 - row.characteristic) % 2 != 0:
            raise ValueError(
                "orientable genus requires an even Euler characteristic per component"
            )
        if g < 0:
            raise ValueError("orientable genus must be nonnegative")
        component_genera.append(g)
        total += g
    return OrientableGenusResult.model_construct(
        per_component=tuple(component_genera), total=total
    )


def orientation_reverse(
    map_: FiniteCombinatorialMap,
) -> OrientationReverseResult:
    """Reverse every local cyclic order.

    Returns the resulting combinatorial map together with the induced bijection
    on faces (``old face index -> new face index``).
    """
    old_walks, old_face_of_dart, _, _ = _face_orbits(map_)
    reversed_rotations = tuple(tuple(reversed(row)) for row in map_.rotations)
    reversed_map = FiniteCombinatorialMap(
        vertex_count=map_.vertex_count,
        darts=map_.darts,
        rotations=reversed_rotations,
    )
    new_walks, new_face_of_dart, _ = face_orbit_data(reversed_map)
    # The reversed face permutation is phi' = alpha . phi^-1 . alpha, so the
    # new orbit of a dart is the reversal image of the old orbit: old face O
    # corresponds to the new face containing the reversed darts of O.  Match
    # through the dart correspondence rather than by set equality, which only
    # works for reverse-symmetric maps.
    face_bijection: dict[int, int] = {}
    for dart_index, dart in enumerate(map_.darts):
        old_face = old_face_of_dart[dart[2]]
        new_face = new_face_of_dart[dart_index]
        existing = face_bijection.setdefault(old_face, new_face)
        if existing != new_face:
            raise ValueError("orientation reversal did not induce a face bijection")
    if len(face_bijection) != len(old_walks) or len(
        set(face_bijection.values())
    ) != len(new_walks):
        raise ValueError("orientation reversal did not induce a face bijection")
    return OrientationReverseResult.model_construct(
        bijection=CombinatorialMapBijection(
            source=map_,
            target=reversed_map,
            kind="FACE",
            source_axis=tuple(range(len(old_walks))),
            target_axis=tuple(range(len(new_walks))),
            images=tuple(face_bijection[i] for i in range(len(old_walks))),
        ),
    )


def dual_map(
    map_: FiniteCombinatorialMap,
) -> DualResult:
    """Return the exact embedded dual.

    - one dual vertex per primal face;
    - one dual dart per primal dart;
    - dual reversal inherited from primal reversal;
    - dual tail/head determined by the two incident face sides;
    - dual rotation determined by the cyclic order of darts around each primal
      face boundary (the dual vertex).

    The dual of a bridge becomes a loop.  Parallel dual edges are retained
    with identity.  Returns the dual map and the primal-dart -> dual-dart
    bijection (the identity here, since dual darts inherit primal dart indices).
    """
    walks, face_of_dart, _, _ = _face_orbits(map_)
    n = len(map_.darts)
    face_count = len(walks)
    if face_count == 0:
        raise ValueError("the primal map must have at least one face")
    # Each dual dart inherits its primal dart index. Its tail and head are the
    # primal faces on the two sides of the primal edge: the tail-face is the
    # face containing the dart itself, and the head-face is the face
    # containing the dart's reverse.
    dual_darts: list[tuple[int, int, int]] = []
    for dart_index in range(n):
        tail_face = face_of_dart[dart_index]
        reverse = map_.darts[dart_index][2]
        head_face = face_of_dart[reverse]
        dual_darts.append((tail_face, head_face, reverse))
    # The dual rotation at dual vertex f is the cyclic order of dual darts
    # whose tail-face is f -- i.e. the darts on the boundary of primal face f.
    # The face walk is already in cyclic order, so use it directly.
    face_darts: dict[int, list[int]] = {f: [] for f in range(face_count)}
    for face_index, walk in enumerate(walks):
        face_darts[face_index] = list(walk)
    dual_rotations: list[tuple[int, ...]] = []
    for f in range(face_count):
        row = face_darts[f]
        if not row:
            raise ValueError(
                f"primal face {f} has an empty boundary, which is impossible for a valid map"
            )
        dual_rotations.append(tuple(row))
    dual = FiniteCombinatorialMap(
        vertex_count=face_count,
        darts=tuple(dual_darts),
        rotations=tuple(dual_rotations),
    )
    return DualResult.model_construct(
        bijection=CombinatorialMapBijection(
            source=map_,
            target=dual,
            kind="DART",
            source_axis=tuple(range(n)),
            target_axis=tuple(range(n)),
            images=tuple(range(n)),
        )
    )


def vertex_face_incidence(
    map_: FiniteCombinatorialMap,
) -> VertexFaceIncidenceResult:
    """Return the exact finite incidence structure between vertices and faces.

    Returns ``(multiplicity, boolean_incidence)``:
    - ``multiplicity``: ``(vertex, face) -> count`` where ``count`` is the
      number of times the vertex occurs on the facial boundary.
    - ``boolean_incidence``: ``vertex -> set of incident face indices``
    """
    source = face_orbits(map_)
    walks = source.face_walks
    multiplicity: dict[tuple[int, int], int] = {}
    for face_index, walk in enumerate(walks):
        for dart in walk:
            vertex = map_.darts[dart][0]
            key = (vertex, face_index)
            multiplicity[key] = multiplicity.get(key, 0) + 1
    return VertexFaceIncidenceResult.model_construct(
        source=source,
        multiplicity=SparseRationalMatrix(
            row_count=map_.vertex_count,
            entries=tuple(
                SparseRationalMatrixEntry(
                    row=vertex,
                    column=face,
                    value=CanonicalRational.from_integer_ratio(count, 1),
                )
                for (vertex, face), count in sorted(multiplicity.items())
            ),
            column_count=len(walks),
        ),
    )


def _admit_embedding_candidate(
    graph: SimpleUndirectedGraph, rotations: tuple[tuple[int, ...], ...]
) -> None:
    """Share the catalog checker envelope with native callers."""

    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.graph_carrier",
            message="graph must be a SimpleUndirectedGraph",
        )
    if type(rotations) is not tuple or any(
        type(row) is not tuple or any(type(edge) is not int for edge in row)
        for row in rotations
    ):
        raise OperationDomainValidationError(
            location=("rotations",),
            code="topology.embedding.rotations_shape",
            message="rotations must be a tuple of tuple rows containing edge indices",
        )
    if not graph.vertices:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.empty_graph",
            message="the embedding checker requires at least one graph vertex",
        )
    if len(graph.vertices) > MAX_EMBEDDING_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="topology.embedding.vertex_bound",
            message="graph vertices exceed the admitted embedding-check envelope",
        )
    if len(graph.edges) > MAX_EMBEDDING_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="topology.embedding.edge_bound",
            message="graph edges exceed the admitted embedding-check envelope",
        )
    if any(len(row) > MAX_EMBEDDING_DEGREE for row in rotations):
        raise OperationResourceAdmissionError(
            location=("rotations",),
            code="topology.embedding.degree_bound",
            message="a local rotation exceeds the admitted embedding-check envelope",
        )
    if sum(len(row) for row in rotations) > MAX_EMBEDDING_ROTATION_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("rotations",),
            code="topology.embedding.rotation_bound",
            message="rotation entries exceed the admitted embedding-check envelope",
        )


def _embedding_adjacency(
    graph: SimpleUndirectedGraph,
) -> tuple[dict[str, int], list[set[int]]]:
    index = {label: position for position, label in enumerate(graph.vertices)}
    incident: list[set[int]] = [set() for _ in graph.vertices]
    for edge_index, (left, right) in enumerate(graph.edges):
        incident[index[left]].add(edge_index)
        incident[index[right]].add(edge_index)
    return index, incident


def _embedding_connected(graph: SimpleUndirectedGraph, index: dict[str, int]) -> bool:
    parent = list(range(len(graph.vertices)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in graph.edges:
        first, second = find(index[left]), find(index[right])
        if first != second:
            parent[first] = second
    return len({find(node) for node in range(len(graph.vertices))}) <= 1


def _canonical_edge_rotation(row: tuple[int, ...]) -> tuple[int, ...]:
    if not row:
        return ()
    pivot = row.index(min(row))
    return row[pivot:] + row[:pivot]


def _invalid_embedding(
    graph: SimpleUndirectedGraph,
    rotations: tuple[tuple[int, ...], ...],
    code: str,
    detail: str,
) -> OrientableEmbeddingCheckResult:
    canonical = tuple(_canonical_edge_rotation(tuple(row)) for row in rotations)
    return OrientableEmbeddingCheckResult._from_kernel(
        graph=graph,
        status="INVALID_EMBEDDING",
        rotations=canonical,
        vertices=len(graph.vertices),
        edges=len(graph.edges),
        obstruction_code=code,
        obstruction_detail=detail,
    )


def _embedding_obstruction(
    graph: SimpleUndirectedGraph,
    rotations: tuple[tuple[int, ...], ...],
    incident: list[set[int]],
) -> tuple[str, str] | None:
    """Return the first malformed-incidence obstruction, if any."""

    if len(rotations) != len(graph.vertices):
        return (
            "ROTATION_ROW_COUNT",
            "the rotation system must carry exactly one row per graph vertex",
        )
    for position, row in enumerate(rotations):
        if len(row) != len(incident[position]):
            return (
                "ROTATION_DEGREE_MISMATCH",
                f"rotation row {position} does not list every incident edge",
            )
    for position, row in enumerate(rotations):
        seen: set[int] = set()
        for edge_index in row:
            if not 0 <= edge_index < len(graph.edges):
                return (
                    "EDGE_INDEX_OUT_OF_RANGE",
                    f"rotation row {position} references an undeclared edge index",
                )
            if edge_index not in incident[position]:
                return (
                    "FOREIGN_INCIDENCE",
                    f"edge {edge_index} is not incident to vertex {position}",
                )
            if edge_index in seen:
                return (
                    "DUPLICATE_INCIDENCE",
                    f"rotation row {position} repeats edge {edge_index}",
                )
            seen.add(edge_index)
    return None


def _embedding_ledger(
    graph: SimpleUndirectedGraph,
    canonical: tuple[tuple[int, ...], ...],
    index: dict[str, int],
) -> OrientableEmbeddingCheckResult:
    """Build the complete dart-permutation and face ledger of an admitted system."""

    vertex_count = len(graph.vertices)
    endpoints = [(index[left], index[right]) for left, right in graph.edges]
    darts: list[tuple[int, int, int]] = []
    for edge_index, (tail, head) in enumerate(endpoints):
        darts.append((tail, head, 2 * edge_index + 1))
        darts.append((head, tail, 2 * edge_index))
    dart_rotations: list[tuple[int, ...]] = []
    for position in range(vertex_count):
        row = []
        for edge_index in canonical[position]:
            tail, _head = endpoints[edge_index]
            row.append(2 * edge_index if tail == position else 2 * edge_index + 1)
        dart_rotations.append(tuple(row))
    dart_count = len(darts)
    alpha = [dart[2] for dart in darts]
    sigma = [0] * dart_count
    for rotation in dart_rotations:
        for offset, dart in enumerate(rotation):
            sigma[dart] = rotation[(offset + 1) % len(rotation)]
    phi = [alpha[sigma[dart]] for dart in range(dart_count)]
    face_walks: list[tuple[int, ...]] = []
    face_of_dart = [0] * dart_count
    visited = [False] * dart_count
    for start in range(dart_count):
        if visited[start]:
            continue
        walk: list[int] = []
        current = start
        while not visited[current]:
            visited[current] = True
            face_of_dart[current] = len(face_walks)
            walk.append(current)
            current = phi[current]
        face_walks.append(tuple(walk))
    faces = len(face_walks) if dart_count else 1
    characteristic = vertex_count - len(graph.edges) + faces
    excess = 2 - characteristic
    if excess < 0 or excess % 2 != 0:
        raise RuntimeError(
            "a connected unsigned rotation system must induce an orientable surface"
        )
    return OrientableEmbeddingCheckResult._from_kernel(
        graph=graph,
        status="ORIENTABLE_CELLULAR_EMBEDDING",
        rotations=canonical,
        dart_rotations=tuple(dart_rotations),
        darts=tuple(darts),
        alpha=tuple(alpha),
        sigma=tuple(sigma),
        phi=tuple(phi),
        face_walks=tuple(face_walks),
        face_of_dart=tuple(face_of_dart),
        vertices=vertex_count,
        edges=len(graph.edges),
        faces=faces,
        euler_characteristic=characteristic,
        genus=excess // 2,
    )


def check_orientable_embedding(
    graph: SimpleUndirectedGraph, rotations: tuple[tuple[int, ...], ...]
) -> OrientableEmbeddingCheckResult:
    """Check one supplied rotation system as an orientable cellular embedding.

    The rotation system is a cyclic order of incident edge indices at every
    vertex.  Every accepted unsigned rotation system describes a cellular
    embedding of the connected graph in a closed orientable surface, so
    orientability is a convention here rather than an optional flag: faces are
    the cycles of ``phi = alpha . sigma``.  This is a checker for a supplied
    rotation system, not a genus minimizer.
    """

    _admit_embedding_candidate(graph, rotations)
    index, incident = _embedding_adjacency(graph)
    obstruction = _embedding_obstruction(graph, rotations, incident)
    if obstruction is not None:
        return _invalid_embedding(graph, rotations, *obstruction)
    if not _embedding_connected(graph, index):
        return _invalid_embedding(
            graph,
            rotations,
            "GRAPH_DISCONNECTED",
            "the supplied graph is not connected",
        )
    canonical = tuple(_canonical_edge_rotation(tuple(row)) for row in rotations)
    return _embedding_ledger(graph, canonical, index)


def verify_orientable_embedding(claim: OrientableEmbeddingCheckResult) -> bool:
    """Check a claimed embedding by replaying the supplied rotation system."""
    return check_orientable_embedding(claim.graph, claim.rotations) == claim


def _rotation_system_row_choices(
    incident: list[set[int]],
) -> list[Callable[[], Iterator[tuple[int, ...]]]]:
    """Return one lazy cyclic-order factory per vertex.

    Fixing the first entry to the least incident edge identifies cyclic
    shifts, so each factory yields exactly ``(degree - 1)!`` orders (one
    order for an isolated or degree-one vertex) in lexicographic order.
    Returning factories, not materialized lists, keeps a high-degree vertex
    from building its factorial many rows before the candidate budget is
    consulted.
    """

    factories: list[Callable[[], Iterator[tuple[int, ...]]]] = []
    for edges in incident:
        ordered = sorted(edges)
        if not ordered:
            factories.append(lambda: iter(((),)))
            continue
        first, rest = ordered[0], ordered[1:]

        def factory(
            first: int = first, rest: list[int] = rest
        ) -> Iterator[tuple[int, ...]]:
            return ((first, *perm) for perm in permutations(rest))

        factories.append(factory)
    return factories


def _rotation_system_total(incident: list[set[int]]) -> int:
    """Return the exact rotation-system count ``product of (degree - 1)!``."""

    total = 1
    for edges in incident:
        if edges:
            total *= factorial(len(edges) - 1)
    return total


def _lazy_product(
    factories: list[Callable[[], Iterator[tuple[int, ...]]]],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Cartesian product that consumes each factory only as needed."""

    if not factories:
        yield ()
        return
    first, rest = factories[0], factories[1:]
    for head in first():
        for tail in _lazy_product(rest):
            yield (head, *tail)


def _admit_rotation_system_search(
    graph: SimpleUndirectedGraph, max_genus: int, max_candidates: int
) -> None:
    """Share the genus-search envelope with native callers."""

    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.graph_carrier",
            message="graph must be a SimpleUndirectedGraph",
        )
    if not graph.vertices:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.empty_graph",
            message="the genus search requires at least one graph vertex",
        )
    if type(max_genus) is not int or max_genus < 0:
        raise OperationDomainValidationError(
            location=("max_genus",),
            code="topology.embedding.genus_bound",
            message="the genus search requires a nonnegative genus bound",
        )
    if type(max_candidates) is not int or max_candidates < 1:
        raise OperationDomainValidationError(
            location=("max_candidates",),
            code="topology.embedding.candidate_budget",
            message="the genus search requires a positive candidate budget",
        )
    if max_candidates > MAX_ROTATION_SYSTEM_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("max_candidates",),
            code="topology.embedding.candidate_bound",
            message="the candidate budget exceeds the admitted search envelope",
        )
    _admit_embedding_candidate(graph, tuple(() for _ in graph.vertices))


def _unknown_rotation_system(
    graph: SimpleUndirectedGraph,
    max_genus: int,
    max_candidates: int,
    total: int,
    examined: int,
    code: str,
    detail: str,
) -> RotationSystemFindResult:
    return RotationSystemFindResult._from_kernel(
        graph=graph,
        max_genus=max_genus,
        max_candidates=max_candidates,
        status="UNKNOWN",
        candidates_examined=examined,
        total_candidates=total,
        reason=code,
        reason_detail=detail,
    )


def find_rotation_system(
    graph: SimpleUndirectedGraph, max_genus: int, max_candidates: int
) -> RotationSystemFindResult:
    """Find a rotation system of genus at most ``max_genus`` by bounded search.

    Rotation systems are enumerated in deterministic vertex order with each
    local row ranging over cyclic orders (first entry fixed), so every
    distinct cellular embedding is examined exactly once up to cyclic
    shifts.  Each candidate is decided by the exact orientable checker:

    - ``FOUND`` carries the first system of genus at most ``max_genus``
      with its checker certificate;
    - ``EXHAUSTED`` carries the receipt that every one of the
      ``total_candidates`` systems was examined and none qualified;
    - ``UNKNOWN`` carries a bounded reason: the candidate budget ran out
      before exhaustion, or the graph is disconnected (outside the
      connected Euler convention).

    A truncated search never yields a negative conclusion: ``EXHAUSTED``
    requires examining every rotation system within budget.
    """

    _admit_rotation_system_search(graph, max_genus, max_candidates)
    index, incident = _embedding_adjacency(graph)
    rows = _rotation_system_row_choices(incident)
    total = _rotation_system_total(incident)
    if not _embedding_connected(graph, index):
        return _unknown_rotation_system(
            graph,
            max_genus,
            max_candidates,
            total,
            0,
            "GRAPH_DISCONNECTED",
            "the genus search requires a connected graph",
        )
    examined = 0
    for rotations in _lazy_product(rows):
        if examined >= max_candidates:
            return _unknown_rotation_system(
                graph,
                max_genus,
                max_candidates,
                total,
                examined,
                "CANDIDATE_BUDGET_EXCEEDED",
                "the candidate budget ran out before exhaustion",
            )
        certificate = check_orientable_embedding(graph, rotations)
        if certificate.status != "ORIENTABLE_CELLULAR_EMBEDDING":
            raise RuntimeError(
                "an enumerated rotation system must check as a cellular embedding"
            )
        examined += 1
        if certificate.genus <= max_genus:
            return RotationSystemFindResult._from_kernel(
                graph=graph,
                max_genus=max_genus,
                max_candidates=max_candidates,
                status="FOUND",
                rotations=rotations,
                certificate=certificate,
                candidates_examined=examined,
                total_candidates=total,
            )
    return RotationSystemFindResult._from_kernel(
        graph=graph,
        max_genus=max_genus,
        max_candidates=max_candidates,
        status="EXHAUSTED",
        candidates_examined=examined,
        total_candidates=total,
    )


def verify_rotation_system_find(claim: RotationSystemFindResult) -> bool:
    """Check a claimed genus search by replaying it within its budget."""
    return (
        find_rotation_system(claim.graph, claim.max_genus, claim.max_candidates)
        == claim
    )


def _signed_embedding_signs(
    graph: SimpleUndirectedGraph,
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...] | None,
    twisted_edges: tuple[int, ...] | None,
) -> tuple[int, ...] | SignedEmbeddingCheckResult:
    """Resolve the two sign encodings to one admission-checked sign tuple."""

    edges = graph.edges
    if signs is not None and twisted_edges is not None:
        return _invalid_signed_embedding(
            graph,
            rotations,
            signs,
            "SIGN_INDEX_OUT_OF_RANGE",
            "provide either signs or twisted_edges, not both",
            twisted_edges=twisted_edges,
        )
    if signs is not None:
        if any(sign not in (0, 1) for sign in signs):
            return _invalid_signed_embedding(
                graph,
                rotations,
                signs,
                "SIGN_INDEX_OUT_OF_RANGE",
                "each edge sign must be 0 (twisted) or 1 (untwisted)",
            )
        if len(signs) != len(edges):
            return _invalid_signed_embedding(
                graph,
                rotations,
                signs,
                "SIGN_INDEX_OUT_OF_RANGE",
                "signs must carry exactly one entry per graph edge",
            )
        return signs
    if twisted_edges is None:
        return (1,) * len(edges)
    seen: set[int] = set()
    for edge_index in twisted_edges:
        if type(edge_index) is not int or not 0 <= edge_index < len(edges):
            return _invalid_signed_embedding(
                graph,
                rotations,
                None,
                "SIGN_INDEX_OUT_OF_RANGE",
                "twisted_edges must index a declared graph edge",
                twisted_edges=twisted_edges,
            )
        if edge_index in seen:
            return _invalid_signed_embedding(
                graph,
                rotations,
                None,
                "SIGN_INDEX_OUT_OF_RANGE",
                f"twisted_edges repeats edge {edge_index}",
                twisted_edges=twisted_edges,
            )
        seen.add(edge_index)
    return tuple(0 if edge_index in seen else 1 for edge_index in range(len(edges)))


def _invalid_signed_embedding(
    graph: SimpleUndirectedGraph,
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...] | None,
    code: str,
    detail: str,
    twisted_edges: tuple[int, ...] | None = None,
) -> SignedEmbeddingCheckResult:
    canonical = tuple(_canonical_edge_rotation(tuple(row)) for row in rotations)
    return SignedEmbeddingCheckResult._from_kernel(
        graph=graph,
        status="INVALID_EMBEDDING",
        signs=signs,
        twisted_edges=twisted_edges,
        rotations=canonical,
        vertices=len(graph.vertices),
        edges=len(graph.edges),
        obstruction_code=code,
        obstruction_detail=detail,
    )


def _signed_cover_faces(
    endpoints: list[tuple[int, int]],
    canonical: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Project the double-cover faces to base dart walks."""
    return signed_projected_faces(endpoints, canonical, signs)


def _signed_base_dart_data(
    endpoints: list[tuple[int, int]],
    canonical: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Build base darts, dart rotations, alpha, and sigma of an admitted system."""

    darts: list[tuple[int, int, int]] = []
    for edge_index, (tail, head) in enumerate(endpoints):
        darts.append((tail, head, 2 * edge_index + 1))
        darts.append((head, tail, 2 * edge_index))
    dart_rotations: list[tuple[int, ...]] = []
    for position in range(len(canonical)):
        row = []
        for edge_index in canonical[position]:
            tail, _head = endpoints[edge_index]
            row.append(2 * edge_index if tail == position else 2 * edge_index + 1)
        dart_rotations.append(tuple(row))
    alpha = [dart[2] for dart in darts]
    sigma = [0] * len(darts)
    for rotation in dart_rotations:
        for offset, dart in enumerate(rotation):
            sigma[dart] = rotation[(offset + 1) % len(rotation)]
    return tuple(darts), tuple(dart_rotations), tuple(alpha), tuple(sigma)


def _require_closed_odd_witness(
    darts: tuple[tuple[int, int, int], ...],
    signs: tuple[int, ...],
    witness: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Require a nonempty closed odd-twist dart walk."""

    dart_count = len(darts)
    if witness is None or not witness:
        raise RuntimeError("the odd-twist witness must be a closed odd walk")
    tails = [darts[dart][0] for dart in witness]
    heads = [darts[dart][1] for dart in witness]
    if (
        any(not 0 <= dart < dart_count for dart in witness)
        or any(
            heads[position] != tails[(position + 1) % len(witness)]
            for position in range(len(witness))
        )
        or sum(1 for dart in witness if signs[dart // 2] == 0) % 2 != 1
    ):
        raise RuntimeError("the odd-twist witness must be a closed odd walk")
    return witness


def _canonical_walk_key(walk: tuple[int, ...]) -> tuple[int, ...]:
    if not walk:
        return ()
    return min(walk[offset:] + walk[:offset] for offset in range(len(walk)))


def _reverse_walk_key(walk: tuple[int, ...]) -> tuple[int, ...]:
    if not walk:
        return ()
    reversed_walk = tuple(dart ^ 1 for dart in reversed(walk))
    return min(
        reversed_walk[offset:] + reversed_walk[:offset]
        for offset in range(len(reversed_walk))
    )


def _signed_base_faces(projected: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Quotient projected cover walks to one walk per base face.

    Exact-duplicate walks (the same directed walk lifted twice) pair
    first; remaining walks pair with their reverse-direction partner.
    A walk equal to its own reverse is kept alone.  Selection is
    deterministic: groups are processed in canonical-key order.
    """

    keys = [_canonical_walk_key(walk) for walk in projected]
    reverse_keys = [_reverse_walk_key(walk) for walk in projected]
    used = [False] * len(projected)
    base: list[tuple[int, ...]] = []
    groups: dict[tuple[int, ...], list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    leftovers: list[int] = []
    for key in sorted(groups):
        members = groups[key]
        for offset in range(0, len(members) - 1, 2):
            used[members[offset]] = used[members[offset + 1]] = True
            base.append(projected[members[offset]])
        if len(members) % 2 == 1:
            leftovers.append(members[-1])
    leftovers.sort(key=lambda position: keys[position])
    while leftovers:
        position = leftovers.pop()
        if used[position]:
            continue
        partner: int | None = None
        for candidate in leftovers:
            if used[candidate] or len(projected[candidate]) != len(projected[position]):
                continue
            if (
                keys[candidate] == keys[position]
                or keys[candidate] == reverse_keys[position]
            ):
                partner = candidate
                break
        if partner is None:
            if reverse_keys[position] == keys[position]:
                used[position] = True
                base.append(projected[position])
                continue
            raise RuntimeError("a projected cover walk has no base-face partner")
        leftovers.remove(partner)
        used[position] = used[partner] = True
        first, second = keys[position], keys[partner]
        base.append(projected[position] if first <= second else projected[partner])
    base.sort(key=_canonical_walk_key)
    return base


def _signed_balance_witness(
    endpoints: list[tuple[int, int]],
    vertex_count: int,
    signs: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return an odd-twist closed dart walk, or None when balanced."""

    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in range(vertex_count)]
    for edge_index, (left, right) in enumerate(endpoints):
        parity = 1 if signs[edge_index] == 0 else 0
        adjacency[left].append((right, parity, edge_index))
        adjacency[right].append((left, parity, edge_index))
    potential: dict[int, int] = {}
    parent: dict[int, tuple[int, int]] = {}
    for root in range(vertex_count):
        if root in potential:
            continue
        potential[root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for target, parity, edge_index in adjacency[node]:
                if target not in potential:
                    potential[target] = potential[node] ^ parity
                    parent[target] = (node, edge_index)
                    queue.append(target)
                elif potential[target] != potential[node] ^ parity:
                    chain: list[int] = [node]
                    cursor = node
                    while cursor != root:
                        cursor = parent[cursor][0]
                        chain.append(cursor)
                    down: list[int] = [target]
                    cursor = target
                    while cursor != root:
                        cursor = parent[cursor][0]
                        down.append(cursor)
                    down = down[::-1]
                    vertex_walk = chain + down[1:] + [node]
                    darts: list[int] = []
                    for first, second in pairwise(vertex_walk):
                        for candidate, (left, right) in enumerate(endpoints):
                            if {left, right} == {first, second}:
                                darts.append(
                                    2 * candidate
                                    if left == first
                                    else 2 * candidate + 1
                                )
                                break
                    return tuple(darts)
    return None


def _signed_embedding_ledger(
    graph: SimpleUndirectedGraph,
    signs: tuple[int, ...],
    canonical: tuple[tuple[int, ...], ...],
    index: dict[str, int],
) -> SignedEmbeddingCheckResult:
    """Build the complete signed dart ledger and decide orientability.

    Faces are projected from the orientable double cover (mirrored
    rotations on the second sheet) through the unsigned face ledger, so
    every projected face crosses an even number of twisted edges.
    Orientability is decided by the exact balance test: the surface is
    orientable exactly when every cycle carries an even number of twisted
    edges.  A nonorientable result carries a concrete odd-twist cycle as
    its orientation-reversing witness.
    """

    vertex_count = len(graph.vertices)
    endpoints = [(index[left], index[right]) for left, right in graph.edges]
    darts, dart_rotations, alpha, sigma = _signed_base_dart_data(endpoints, canonical)
    dart_count = len(darts)
    if dart_count == 0:
        if len(signs) != 0:
            raise RuntimeError("signs must be empty for the edgeless embedding")
        return SignedEmbeddingCheckResult._from_kernel(
            graph=graph,
            status="ORIENTABLE_EMBEDDING",
            signs=signs,
            rotations=canonical,
            dart_rotations=dart_rotations,
            darts=darts,
            alpha=alpha,
            sigma=sigma,
            face_walks=(),
            vertices=vertex_count,
            edges=len(graph.edges),
            faces=1,
            euler_characteristic=2,
            genus=0,
            orientable=True,
            witness_dart_walk=None,
        )
    projected = _signed_cover_faces(endpoints, canonical, signs)
    faces = _signed_base_faces(projected)
    occurrences = [0] * len(graph.edges)
    for walk in faces:
        for dart in walk:
            if not 0 <= dart < dart_count:
                raise RuntimeError("a projected face references no base dart")
            occurrences[dart // 2] += 1
    if any(count != 2 for count in occurrences):
        raise RuntimeError("every base edge must occur in two face positions")
    characteristic = vertex_count - len(graph.edges) + len(faces)
    witness = _signed_balance_witness(endpoints, vertex_count, signs)
    if witness is None:
        excess = 2 - characteristic
        if excess < 0 or excess % 2 != 0:
            raise RuntimeError(
                "a balanced signed rotation system must induce an orientable surface"
            )
        return SignedEmbeddingCheckResult._from_kernel(
            graph=graph,
            status="ORIENTABLE_EMBEDDING",
            signs=signs,
            rotations=canonical,
            dart_rotations=dart_rotations,
            darts=darts,
            alpha=alpha,
            sigma=sigma,
            face_walks=tuple(faces),
            vertices=vertex_count,
            edges=len(graph.edges),
            faces=len(faces),
            euler_characteristic=characteristic,
            genus=excess // 2,
            orientable=True,
            witness_dart_walk=None,
        )
    if characteristic > 1:
        raise RuntimeError(
            "an unbalanced signed rotation system must induce a nonorientable surface"
        )
    return SignedEmbeddingCheckResult._from_kernel(
        graph=graph,
        status="NONORIENTABLE_EMBEDDING",
        signs=signs,
        rotations=canonical,
        dart_rotations=dart_rotations,
        darts=darts,
        alpha=alpha,
        sigma=sigma,
        face_walks=tuple(faces),
        vertices=vertex_count,
        edges=len(graph.edges),
        faces=len(faces),
        euler_characteristic=characteristic,
        genus=2 - characteristic,
        orientable=False,
        witness_dart_walk=_require_closed_odd_witness(darts, signs, witness),
    )


def _admit_signed_embedding_candidate(
    graph: SimpleUndirectedGraph, rotations: tuple[tuple[int, ...], ...]
) -> None:
    """Share the signed checker envelope with native callers."""

    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.graph_carrier",
            message="graph must be a SimpleUndirectedGraph",
        )
    if type(rotations) is not tuple or any(
        type(row) is not tuple or any(type(edge) is not int for edge in row)
        for row in rotations
    ):
        raise OperationDomainValidationError(
            location=("rotations",),
            code="topology.embedding.rotations_shape",
            message="rotations must be a tuple of tuple rows containing edge indices",
        )
    if not graph.vertices:
        raise OperationDomainValidationError(
            location=("graph",),
            code="topology.embedding.empty_graph",
            message="the embedding checker requires at least one graph vertex",
        )
    if len(graph.vertices) > MAX_EMBEDDING_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="topology.embedding.vertex_bound",
            message="graph vertices exceed the admitted embedding-check envelope",
        )
    if len(graph.edges) > MAX_EMBEDDING_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="topology.embedding.edge_bound",
            message="graph edges exceed the admitted embedding-check envelope",
        )
    if any(len(row) > MAX_EMBEDDING_DEGREE for row in rotations):
        raise OperationResourceAdmissionError(
            location=("rotations",),
            code="topology.embedding.degree_bound",
            message="a local rotation exceeds the admitted embedding-check envelope",
        )
    if sum(len(row) for row in rotations) > MAX_EMBEDDING_ROTATION_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("rotations",),
            code="topology.embedding.rotation_bound",
            message="rotation entries exceed the admitted embedding-check envelope",
        )


def check_signed_embedding(
    graph: SimpleUndirectedGraph,
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...] | None = None,
    twisted_edges: tuple[int, ...] | None = None,
) -> SignedEmbeddingCheckResult:
    """Check one supplied signed rotation system for orientability.

    A signed rotation system gives every edge a sign: 1 (untwisted) or 0
    (twisted, orientation-reversing).  Faces are projected from the
    orientable double cover (mirrored rotations on the second sheet)
    through the unsigned face ledger, so every projected face crosses an
    even number of twisted edges.  Orientability is decided by the exact
    balance test: the surface is orientable exactly when every cycle
    carries an even number of twisted edges.  A nonorientable result
    carries a concrete odd-twist cycle as its orientation-reversing
    witness.  The surface classification is ``chi = V - E + F`` with
    orientable genus ``g = (2 - chi) / 2`` or nonorientable genus
    ``h = 2 - chi``.  This is a checker for a supplied signed rotation
    system, not a genus minimizer.
    """

    _admit_signed_embedding_candidate(graph, rotations)
    resolved = _signed_embedding_signs(graph, rotations, signs, twisted_edges)
    if isinstance(resolved, SignedEmbeddingCheckResult):
        return resolved
    index, incident = _embedding_adjacency(graph)
    obstruction = _embedding_obstruction(graph, rotations, incident)
    if obstruction is not None:
        return _invalid_signed_embedding(graph, rotations, resolved, *obstruction)
    if not _embedding_connected(graph, index):
        return _invalid_signed_embedding(
            graph,
            rotations,
            resolved,
            "GRAPH_DISCONNECTED",
            "the supplied graph is not connected",
        )
    canonical = tuple(_canonical_edge_rotation(tuple(row)) for row in rotations)
    return _signed_embedding_ledger(graph, resolved, canonical, index)


def verify_signed_embedding(claim: SignedEmbeddingCheckResult) -> bool:
    """Check a claimed signed embedding by replaying its exact encoding.

    Both the resolved ``signs`` and any retained ``twisted_edges`` are
    replayed, so encoding-error results round-trip as well as admitted ones.
    """
    return (
        check_signed_embedding(
            claim.graph,
            claim.rotations,
            signs=claim.signs,
            twisted_edges=claim.twisted_edges,
        )
        == claim
    )


def verify_dual(claim: DualResult) -> bool:
    """Check embedded duality and its declared canonical dart bijection."""
    return dual_map(claim.bijection.source) == claim


def verify_orientation_reverse(claim: OrientationReverseResult) -> bool:
    """Check reversal and the induced face bijection after map admission."""
    return orientation_reverse(claim.bijection.source) == claim


def verify_vertex_face_incidence(claim: VertexFaceIncidenceResult) -> bool:
    """Check the face axis and its incidence multiplicities against the map."""
    return vertex_face_incidence(claim.source.map) == claim
