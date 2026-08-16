import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _norm_edges(orbits):
    if not isinstance(orbits, list):
        return None
    norm = []
    for orbit in orbits:
        if not isinstance(orbit, list):
            return None
        normalized_orbit = []
        for edge in orbit:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or any(not isinstance(vertex, str) for vertex in edge)
            ):
                return None
            normalized_orbit.append(sorted(edge))
        norm.append(sorted(normalized_orbit))
    return sorted(norm)


def _norm_vertices(orbits):
    if not isinstance(orbits, list) or any(
        not isinstance(orbit, list)
        or any(not isinstance(vertex, str) for vertex in orbit)
        for orbit in orbits
    ):
        return None
    return sorted([sorted(orbit) for orbit in orbits])


def _orbits(items, generators, action):
    remaining = set(items)
    result = []
    while remaining:
        orbit = {min(remaining)}
        frontier = list(orbit)
        while frontier:
            item = frontier.pop()
            for generator in generators:
                image = action(generator, item)
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        if not orbit <= set(items):
            return None
        remaining -= orbit
        result.append(sorted(orbit))
    return result


def _expected_orbits(x):
    graph = x.get("graph")
    declared = x.get("generators")
    if not isinstance(graph, dict) or not isinstance(declared, list):
        return None
    vertices = graph.get("vertices")
    raw_edges = graph.get("edges")
    if (
        not isinstance(vertices, list)
        or any(not isinstance(vertex, str) for vertex in vertices)
        or len(set(vertices)) != len(vertices)
        or not isinstance(raw_edges, list)
    ):
        return None
    edges = []
    for edge in raw_edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(vertex, str) for vertex in edge)
        ):
            return None
        edges.append(tuple(sorted(edge)))
    generators = []
    for declared_generator in declared:
        mapping = (
            declared_generator.get("mapping")
            if isinstance(declared_generator, dict)
            else None
        )
        if (
            not isinstance(mapping, dict)
            or set(mapping) != set(vertices)
            or set(mapping.values()) != set(vertices)
        ):
            return None
        generators.append(mapping)
    vertex_orbits = _orbits(vertices, generators, lambda g, vertex: g[vertex])
    edge_orbits = _orbits(
        edges,
        generators,
        lambda g, edge: tuple(sorted((g[edge[0]], g[edge[1]]))),
    )
    if vertex_orbits is None or edge_orbits is None:
        return None
    return vertex_orbits, [[list(edge) for edge in orbit] for orbit in edge_orbits]


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict):
        return False
    expected = _expected_orbits(x)
    if expected is None:
        return False
    expected_vertices, expected_edges = expected
    return _norm_vertices(result.get("vertex_orbits")) == _norm_vertices(
        expected_vertices
    ) and _norm_edges(result.get("edge_orbits")) == _norm_edges(expected_edges)


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = _math(s, x) if protocol_ok else False
    reward = float(protocol_ok and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
