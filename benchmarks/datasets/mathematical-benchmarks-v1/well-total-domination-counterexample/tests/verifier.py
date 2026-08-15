import json
import math
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
FIXTURE_NAME = (
    "mathematical-benchmarks-v1-well-total-domination-counterexample-input.json"
)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def graph_from_fixture(fixture):
    graph = fixture["graph"]
    raw_vertices = graph["vertices"]
    if not isinstance(raw_vertices, list) or not raw_vertices:
        raise ValueError("invalid vertices")
    vertices = {str(vertex) for vertex in raw_vertices}
    if len(vertices) != len(raw_vertices):
        raise ValueError("duplicate vertices")
    adjacency = {vertex: set() for vertex in vertices}
    seen_edges = set()
    edges = graph["edges"]
    if not isinstance(edges, list):
        raise ValueError("invalid edges")
    for raw_edge in edges:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError("malformed edge")
        left, right = map(str, raw_edge)
        edge = tuple(sorted((left, right)))
        if left == right or left not in vertices or right not in vertices:
            raise ValueError("invalid edge")
        if edge in seen_edges:
            raise ValueError("duplicate edge")
        seen_edges.add(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return vertices, adjacency


def connected(vertices, adjacency):
    todo = [next(iter(vertices))]
    reached = set()
    while todo:
        vertex = todo.pop()
        if vertex in reached:
            continue
        reached.add(vertex)
        todo.extend(adjacency[vertex] - reached)
    return reached == vertices


def is_total_dominating(candidate, vertices, adjacency):
    return all(adjacency[vertex] & candidate for vertex in vertices)


def is_minimal_total_dominating(candidate, vertices, adjacency):
    return is_total_dominating(candidate, vertices, adjacency) and all(
        not is_total_dominating(candidate - {vertex}, vertices, adjacency)
        for vertex in candidate
    )


def result_valid(result, fixture):
    required = {
        "connected",
        "degree_sum",
        "average_degree",
        "pendant_vertices",
        "hypothesis_holds",
        "minimal_total_dominating_sets",
        "well_totally_dominated",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    try:
        vertices, adjacency = graph_from_fixture(fixture)
    except (KeyError, TypeError, ValueError):
        return False
    degree_sum = sum(len(adjacency[vertex]) for vertex in vertices)
    divisor = math.gcd(degree_sum, len(vertices))
    reduced_average = {
        "numerator": degree_sum // divisor,
        "denominator": len(vertices) // divisor,
    }
    pendant_vertices = {vertex for vertex in vertices if len(adjacency[vertex]) == 1}
    raw_sets = result["minimal_total_dominating_sets"]
    if not isinstance(raw_sets, list) or len(raw_sets) != 2:
        return False
    candidate_sets = []
    for raw_candidate in raw_sets:
        if not isinstance(raw_candidate, list):
            return False
        candidate = {str(vertex) for vertex in raw_candidate}
        if len(candidate) != len(raw_candidate) or not candidate <= vertices:
            return False
        candidate_sets.append(candidate)
    submitted_pendants = result["pendant_vertices"]
    if not isinstance(submitted_pendants, list):
        return False
    submitted_pendant_set = {str(vertex) for vertex in submitted_pendants}
    hypothesis_holds = len(vertices) > 1 and degree_sum <= len(vertices) * len(
        pendant_vertices
    )
    return bool(
        result["connected"] is True
        and connected(vertices, adjacency)
        and type(result["degree_sum"]) is int
        and result["degree_sum"] == degree_sum
        and result["average_degree"] == reduced_average
        and len(submitted_pendant_set) == len(submitted_pendants)
        and submitted_pendant_set == pendant_vertices
        and result["hypothesis_holds"] is True
        and hypothesis_holds
        and all(
            is_minimal_total_dominating(candidate, vertices, adjacency)
            for candidate in candidate_sets
        )
        and len(candidate_sets[0]) != len(candidate_sets[1])
        and result["well_totally_dominated"] is False
    )


def main():
    submission = load_submission()
    fixture = load_json(E / FIXTURE_NAME)
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(fixture is not None and result_valid(result, fixture))
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
