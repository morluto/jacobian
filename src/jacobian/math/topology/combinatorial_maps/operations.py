"""Exact native kernels over finite combinatorial maps.

All functions are deterministic and complete for accepted values.  They
need no ``UNKNOWN``, timeout-as-mathematics, search budget, or solver
outcome.  They use a small exact permutation/orbit kernel over immutable
dart IDs; no backend embedding object crosses the boundary.
"""

from __future__ import annotations

from itertools import permutations, product

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
    MAX_ROTATION_SYSTEM_CANDIDATES,
    CombinatorialMapBijection,
    ConnectedComponentsResult,
    DualResult,
    EulerCharacteristicCounts,
    EulerCharacteristicResult,
    FacesResult,
    OrientableEmbeddingCheckResult,
    OrientableGenusResult,
    OrientationReverseResult,
    RotationSystemFindResult,
    VertexFaceIncidenceResult,
)
from .values import (
    FiniteCombinatorialMap,
    _build_outgoing,
    _validate_facial_budgets,
    _validate_involution,
    _validate_rotation,
)

__all__ = [
    "check_orientable_embedding",
    "connected_components",
    "connected_components_vertices",
    "dual_map",
    "euler_characteristic",
    "face_orbits",
    "find_rotation_system",
    "orientable_genus",
    "orientation_reverse",
    "rotation_successor",
    "verify_dual",
    "verify_orientable_embedding",
    "verify_orientation_reverse",
    "verify_rotation_system_find",
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
) -> list[list[tuple[int, ...]]]:
    """List each vertex's cyclic orders with the first entry fixed.

    Fixing the first entry to the least incident edge identifies cyclic
    shifts, so each list holds exactly ``(degree - 1)!`` orders (one order
    for an isolated or degree-one vertex) in lexicographic order.
    """

    rows: list[list[tuple[int, ...]]] = []
    for edges in incident:
        ordered = sorted(edges)
        if not ordered:
            rows.append([()])
            continue
        first, rest = ordered[0], ordered[1:]
        rows.append([(first, *perm) for perm in permutations(rest)])
    return rows


def _rotation_system_total(rows: list[list[tuple[int, ...]]]) -> int:
    """Return the exact rotation-system count of the row choices."""

    total = 1
    for choices in rows:
        total *= len(choices)
    return total


def _admit_rotation_system_search(
    graph: SimpleUndirectedGraph, max_genus: int, max_candidates: int
) -> None:
    """Share the genus-search envelope with native callers."""

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
    total = _rotation_system_total(rows)
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
    for rotations in product(*rows):
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


def verify_dual(claim: DualResult) -> bool:
    """Check embedded duality and its declared canonical dart bijection."""
    return dual_map(claim.bijection.source) == claim


def verify_orientation_reverse(claim: OrientationReverseResult) -> bool:
    """Check reversal and the induced face bijection after map admission."""
    return orientation_reverse(claim.bijection.source) == claim


def verify_vertex_face_incidence(claim: VertexFaceIncidenceResult) -> bool:
    """Check the face axis and its incidence multiplicities against the map."""
    return vertex_face_incidence(claim.source.map) == claim
