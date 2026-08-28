"""Exact native kernels over finite combinatorial maps.

All functions are deterministic and complete for accepted values.  They
need no ``UNKNOWN``, timeout-as-mathematics, search budget, or solver
outcome.  They use a small exact permutation/orbit kernel over immutable
dart IDs; no backend embedding object crosses the boundary.
"""

from __future__ import annotations

from ._face_orbits import face_orbit_data
from ._models import (
    ConnectedComponentsResult,
    DualResult,
    EulerCharacteristicCounts,
    EulerCharacteristicResult,
    FacesResult,
    OrientableGenusResult,
    OrientationReverseResult,
    VertexFaceIncidenceResult,
)
from .values import FiniteCombinatorialMap

__all__ = [
    "connected_components",
    "connected_components_vertices",
    "dual_map",
    "euler_characteristic",
    "face_orbits",
    "orientable_genus",
    "orientation_reverse",
    "rotation_successor",
    "vertex_face_incidence",
]


def rotation_successor(map_: FiniteCombinatorialMap, dart: int) -> int:
    """Return the dart following ``dart`` in its local rotation."""

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
    walks, face_of_dart, successor = face_orbit_data(map_)
    comp_of_vertex = connected_components_vertices(map_)
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


def connected_components_vertices(
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


def connected_components(
    map_: FiniteCombinatorialMap,
) -> ConnectedComponentsResult:
    """Return the component partition of vertices, darts, and faces."""

    vertex_component = connected_components_vertices(map_)
    walks, _, _, _ = _face_orbits(map_)
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
    vertex_component = connected_components_vertices(map_)
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
    reversed_rotations = tuple(tuple(reversed(row)) for row in map_.rotations)
    reversed_map = FiniteCombinatorialMap(
        vertex_count=map_.vertex_count,
        darts=map_.darts,
        rotations=reversed_rotations,
    )
    old_walks, old_face_of_dart, _, _ = _face_orbits(map_)
    new_walks, new_face_of_dart, _, _ = _face_orbits(reversed_map)
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
        map=map_,
        reversed_map=reversed_map,
        face_bijection=face_bijection,
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
    primal_to_dual = {i: i for i in range(n)}
    return DualResult.model_construct(dual=dual, primal_to_dual=primal_to_dual)


def vertex_face_incidence(
    map_: FiniteCombinatorialMap,
) -> VertexFaceIncidenceResult:
    """Return the exact finite incidence structure between vertices and faces.

    Returns ``(multiplicity, boolean_incidence)``:
    - ``multiplicity``: ``(vertex, face) -> count`` where ``count`` is the
      number of times the vertex occurs on the facial boundary.
    - ``boolean_incidence``: ``vertex -> set of incident face indices``
    """
    walks, _, _, _ = _face_orbits(map_)
    multiplicity: dict[tuple[int, int], int] = {}
    boolean: dict[int, set[int]] = {v: set() for v in range(map_.vertex_count)}
    for face_index, walk in enumerate(walks):
        for dart in walk:
            vertex = map_.darts[dart][0]
            key = (vertex, face_index)
            multiplicity[key] = multiplicity.get(key, 0) + 1
            boolean[vertex].add(face_index)
    nested: dict[int, dict[int, int]] = {}
    for (vertex, face), count in multiplicity.items():
        nested.setdefault(vertex, {})[face] = count
    return VertexFaceIncidenceResult.model_construct(
        multiplicity=nested,
        boolean_incidence={
            vertex: tuple(sorted(boolean[vertex])) for vertex in sorted(boolean)
        },
    )
