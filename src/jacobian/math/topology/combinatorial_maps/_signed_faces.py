"""Pure signed-embedding face ledger used at operation and value boundaries."""

from __future__ import annotations

from collections import deque


def _canonical_cycle(row: tuple[int, ...]) -> tuple[int, ...]:
    if not row:
        return ()
    pivot = row.index(min(row))
    return row[pivot:] + row[:pivot]


def _cover_data(
    endpoints: list[tuple[int, int]],
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    cover_endpoints: list[tuple[int, int]] = []
    for edge_index, (left, right) in enumerate(endpoints):
        turn = 1 if signs[edge_index] == 0 else 0
        cover_endpoints.extend(
            ((left * 2, right * 2 + turn), (left * 2 + 1, right * 2 + 1 - turn))
        )
    cover_rows: list[list[int]] = [[] for _ in range(2 * len(rotations))]
    for vertex, rotation in enumerate(rotations):
        for sheet in (0, 1):
            order = rotation if sheet == 0 else tuple(reversed(rotation))
            row: list[int] = []
            for edge_index in order:
                left, right = endpoints[edge_index]
                turn = 1 if signs[edge_index] == 0 else 0
                if (vertex == left and sheet == 0) or (
                    vertex == right and sheet == turn
                ):
                    row.append(2 * edge_index)
                else:
                    row.append(2 * edge_index + 1)
            cover_rows[2 * vertex + sheet] = row
    return cover_endpoints, cover_rows


def _components(edges: list[tuple[int, int]], vertex_count: int) -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = [False] * vertex_count
    result: list[list[int]] = []
    for start in range(vertex_count):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        component = [start]
        while queue:
            vertex = queue.popleft()
            for target in sorted(adjacency[vertex]):
                if not seen[target]:
                    seen[target] = True
                    queue.append(target)
                    component.append(target)
        result.append(component)
    return result


def _component_faces(
    endpoints: list[tuple[int, int]],
    cover_endpoints: list[tuple[int, int]],
    cover_rows: list[list[int]],
    component: list[int],
) -> list[tuple[int, ...]]:
    members = set(component)
    edge_order = [
        edge_index
        for edge_index, (left, right) in enumerate(cover_endpoints)
        if left in members and right in members
    ]
    local_of = {edge_index: position for position, edge_index in enumerate(edge_order)}
    local_endpoints: list[tuple[int, int]] = []
    for edge_index in edge_order:
        left, right = cover_endpoints[edge_index]
        local_endpoints.append(
            (left, right) if str(left) < str(right) else (right, left)
        )
    local_index = {vertex: position for position, vertex in enumerate(component)}
    darts: list[tuple[int, int, int]] = []
    for edge_index, (left, right) in enumerate(local_endpoints):
        left_position, right_position = local_index[left], local_index[right]
        darts.extend(
            (
                (left_position, right_position, 2 * edge_index + 1),
                (right_position, left_position, 2 * edge_index),
            )
        )
    rows: list[tuple[int, ...]] = []
    for vertex in component:
        row = tuple(
            2 * local_of[edge_index]
            if local_endpoints[local_of[edge_index]][0] == vertex
            else 2 * local_of[edge_index] + 1
            for edge_index in cover_rows[vertex]
            if edge_index in local_of
        )
        rows.append(_canonical_cycle(row))
    alpha = tuple(dart[2] for dart in darts)
    sigma = [0] * len(darts)
    for row in rows:
        for offset, dart in enumerate(row):
            sigma[dart] = row[(offset + 1) % len(row)]
    phi = tuple(alpha[sigma[dart]] for dart in range(len(darts)))
    seen: set[int] = set()
    result: list[tuple[int, ...]] = []
    for start in range(len(darts)):
        if start in seen:
            continue
        walk: list[int] = []
        current = start
        while current not in seen:
            seen.add(current)
            walk.append(current)
            current = phi[current]
        projected: list[int] = []
        for dart in walk:
            local_edge = dart // 2
            global_edge = edge_order[local_edge]
            base_edge = global_edge // 2
            cover_tail = component[darts[dart][0]]
            base_vertex = cover_tail // 2
            base_left, _base_right = endpoints[base_edge]
            projected.append(
                2 * base_edge if base_vertex == base_left else 2 * base_edge + 1
            )
        result.append(tuple(projected))
    return result


def _walk_key(walk: tuple[int, ...]) -> tuple[int, ...]:
    return min(
        (walk[offset:] + walk[:offset] for offset in range(len(walk))),
        default=(),
    )


def _reverse_key(walk: tuple[int, ...]) -> tuple[int, ...]:
    reversed_walk = tuple(dart ^ 1 for dart in reversed(walk))
    return _walk_key(reversed_walk)


def signed_projected_faces(
    endpoints: list[tuple[int, int]],
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Return the projected face walks of the orientable double cover."""
    cover_endpoints, cover_rows = _cover_data(endpoints, rotations, signs)
    projected: list[tuple[int, ...]] = []
    for component in _components(cover_endpoints, 2 * len(rotations)):
        projected.extend(
            _component_faces(endpoints, cover_endpoints, cover_rows, component)
        )
    return projected


def signed_face_walks(
    endpoints: list[tuple[int, int]],
    rotations: tuple[tuple[int, ...], ...],
    signs: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the exact deterministic base-face projection of the double cover."""
    projected = signed_projected_faces(endpoints, rotations, signs)
    keys = [_walk_key(walk) for walk in projected]
    reverse_keys = [_reverse_key(walk) for walk in projected]
    used = [False] * len(projected)
    groups: dict[tuple[int, ...], list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    leftovers: list[int] = []
    result: list[tuple[int, ...]] = []
    for key in sorted(groups):
        members = groups[key]
        for offset in range(0, len(members) - 1, 2):
            used[members[offset]] = used[members[offset + 1]] = True
            result.append(projected[members[offset]])
        if len(members) % 2:
            leftovers.append(members[-1])
    leftovers.sort(key=lambda position: keys[position])
    while leftovers:
        position = leftovers.pop()
        if used[position]:
            continue
        partner = next(
            (
                candidate
                for candidate in leftovers
                if not used[candidate]
                and len(projected[candidate]) == len(projected[position])
                and (
                    keys[candidate] == keys[position]
                    or keys[candidate] == reverse_keys[position]
                )
            ),
            None,
        )
        if partner is None:
            if reverse_keys[position] == keys[position]:
                used[position] = True
                result.append(projected[position])
                continue
            raise ValueError("projected signed face has no deterministic partner")
        leftovers.remove(partner)
        used[position] = used[partner] = True
        result.append(
            projected[position]
            if keys[position] <= keys[partner]
            else projected[partner]
        )
    return tuple(sorted(result, key=_walk_key))


__all__ = ["signed_face_walks", "signed_projected_faces"]
