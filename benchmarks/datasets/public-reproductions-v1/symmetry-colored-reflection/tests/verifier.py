import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
ALLOWED = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})


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


def _math(s, x, e):
    r = s.get("result", {})
    vo = r.get("vertex_orbits")
    eo = r.get("edge_orbits")
    expected = _expected_orbits(x)
    if expected is None:
        return False
    expected_vertices, expected_edges = expected
    return _norm_vertices(vo) == _norm_vertices(expected_vertices) and _norm_edges(
        eo
    ) == _norm_edges(expected_edges)


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    math_correct = _math(s, x, e) if contract else False
    correct = bool(contract and math_correct)
    good = bool(contract and evidence_list_is_bound(s["evidence"]))
    scope = bool(contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(contract and s["claimed_assurance"] == e["maximum_assurance"])
    false = bool(isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED")
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
