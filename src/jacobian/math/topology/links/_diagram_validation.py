"""Finite exact checks for directed planar link-diagram combinatorics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.topology.links._models import LinkCrossing, OrientedDiagramArc


def validate_dart_axes(
    crossings: tuple[LinkCrossing, ...], arcs: tuple[OrientedDiagramArc, ...]
) -> tuple[dict[str, int], dict[str, str], set[str], set[str]]:
    ids = [crossing.crossing_id for crossing in crossings]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("crossing IDs must be unique and strictly ordered")
    dart_owner: dict[str, int] = {}
    for index, crossing in enumerate(crossings):
        if len(set(crossing.half_edges)) != 4:
            raise ValueError("crossing darts must be four distinct IDs")
        if {tuple(sorted(crossing.over_pair)), tuple(sorted(crossing.under_pair))} != {
            (0, 2),
            (1, 3),
        }:
            raise ValueError("over and under strands must be opposite pairs")
        for dart in crossing.half_edges:
            if dart in dart_owner:
                raise ValueError("every dart belongs to exactly one crossing")
            dart_owner[dart] = index
    if 2 * len(arcs) != len(dart_owner):
        raise ValueError("every crossing dart must occur in exactly one arc")
    endpoints: list[str] = []
    partner: dict[str, str] = {}
    tails: set[str] = set()
    heads: set[str] = set()
    for arc in arcs:
        if arc.tail == arc.head:
            raise ValueError("an arc must have distinct endpoint darts")
        endpoints.extend((arc.tail, arc.head))
        if arc.tail in tails or arc.head in heads:
            raise ValueError("each dart has one oriented arc incidence")
        tails.add(arc.tail)
        heads.add(arc.head)
        partner[arc.tail] = arc.head
        partner[arc.head] = arc.tail
    if len(set(endpoints)) != len(endpoints) or set(endpoints) != set(dart_owner):
        raise ValueError("arcs must pair every crossing dart exactly once")
    return dart_owner, partner, tails, heads


def crossing_sign(crossing: LinkCrossing, tails: set[str], heads: set[str]) -> int:
    """Right-hand crossing sign for CCW cyclic positions and directed tangents."""
    radial = ((1, 0), (0, 1), (-1, 0), (0, -1))
    tangents: dict[int, tuple[int, int]] = {}
    for index, dart in enumerate(crossing.half_edges):
        x, y = radial[index]
        if dart in heads:
            x, y = -x, -y
        tangents[index] = (x, y)
    over_x, over_y = tangents[min(crossing.over_pair)]
    under_x, under_y = tangents[min(crossing.under_pair)]
    return 1 if under_x * over_y - under_y * over_x > 0 else -1


def validate_crossing_orientations(
    crossings: tuple[LinkCrossing, ...], tails: set[str], heads: set[str]
) -> None:
    for crossing in crossings:
        for pair in (crossing.over_pair, crossing.under_pair):
            left, right = (crossing.half_edges[index] for index in pair)
            if (left in tails) == (right in tails):
                raise ValueError(
                    "each oriented crossing strand needs one incoming and one outgoing dart"
                )
        if crossing.sign != crossing_sign(crossing, tails, heads):
            raise ValueError(
                "crossing sign disagrees with encoded orientation and cyclic order"
            )


def _permutation_cycles(successor: dict[str, str]) -> list[set[str]]:
    faces: list[set[str]] = []
    unseen = set(successor)
    while unseen:
        start = min(unseen)
        face: set[str] = set()
        cursor = start
        while cursor not in face:
            face.add(cursor)
            unseen.discard(cursor)
            cursor = successor[cursor]
        if cursor != start:
            raise ValueError("face permutation must partition darts into cycles")
        faces.append(face)
    return faces


def _graph_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    unseen = set(adjacency)
    components: list[set[int]] = []
    while unseen:
        component = {min(unseen)}
        stack = list(component)
        unseen.difference_update(component)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def validate_sphere_embedding(
    crossings: tuple[LinkCrossing, ...],
    arcs: tuple[OrientedDiagramArc, ...],
    dart_owner: dict[str, int],
    partner: dict[str, str],
) -> None:
    successor = {
        dart: crossing.half_edges[(index + 1) % 4]
        for crossing in crossings
        for index, dart in enumerate(crossing.half_edges)
    }
    faces = _permutation_cycles({dart: successor[partner[dart]] for dart in successor})
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(crossings))}
    for arc in arcs:
        left, right = dart_owner[arc.tail], dart_owner[arc.head]
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _graph_components(adjacency)
    component_of = {
        vertex: index for index, part in enumerate(components) for vertex in part
    }
    if any(
        len({component_of[dart_owner[dart]] for dart in face}) != 1 for face in faces
    ):
        raise ValueError("face cycle crosses projection components")
    if len(crossings) - len(arcs) + len(faces) != 2 * len(components):
        raise ValueError("crossing rotation system is not an embedding on the sphere")
