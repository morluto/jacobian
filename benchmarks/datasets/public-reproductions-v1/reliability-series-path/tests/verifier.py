import json
from fractions import Fraction
from itertools import product
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _submission_frac(v):
    if not isinstance(v, dict) or set(v) != {"num", "den"}:
        return None
    num, den = (v.get("num"), v.get("den"))
    if type(num) is not int or type(den) is not int or den <= 0:
        return None
    try:
        value = Fraction(num, den)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return value


def _input_frac(v):
    if not isinstance(v, dict) or set(v) != {"num", "den"}:
        return None
    num, den = (v.get("num"), v.get("den"))
    if (
        not isinstance(num, str)
        or not isinstance(den, str)
        or (not num.lstrip("-").isdigit())
        or (not den.isdigit())
    ):
        return None
    try:
        return Fraction(int(num), int(den))
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _graph_parts(x):
    graph = x.get("graph")
    probabilities = x.get("edge_probabilities")
    terminals = x.get("terminals")
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("vertices"), list)
        or (not isinstance(graph.get("edges"), list))
        or (not isinstance(probabilities, list))
        or (not isinstance(terminals, list))
        or (len(terminals) != 2)
    ):
        return None
    return (graph["vertices"], graph["edges"], probabilities, terminals)


def _probabilities_by_edge(probabilities, edges):
    probability_by_edge = {}
    for item in probabilities:
        if not isinstance(item, dict) or set(item) != {"edge", "open_probability"}:
            return None
        edge = item["edge"]
        probability = _input_frac(item["open_probability"])
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(vertex, str) for vertex in edge)
            or (probability is None)
        ):
            return None
        probability_by_edge[frozenset(edge)] = probability
    return probability_by_edge if len(probability_by_edge) == len(edges) else None


def _state_result(edges, state, vertices, terminals, probability_by_edge):
    adjacency = {vertex: set() for vertex in vertices}
    mass = Fraction(1)
    for edge, opened in zip(edges, state, strict=True):
        probability = probability_by_edge.get(frozenset(edge))
        if probability is None:
            return None
        mass *= probability if opened else 1 - probability
        if opened:
            left, right = edge
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached = {terminals[0]}
    frontier = [terminals[0]]
    while frontier:
        vertex = frontier.pop()
        for neighbor in adjacency[vertex] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return (mass, terminals[1] in reached)


def _expected_result(x):
    parts = _graph_parts(x)
    if parts is None:
        return None
    vertices, edges, probabilities, terminals = parts
    probability_by_edge = _probabilities_by_edge(probabilities, edges)
    if probability_by_edge is None:
        return None
    total = Fraction(0)
    for state in product((False, True), repeat=len(edges)):
        state_result = _state_result(
            edges, state, vertices, terminals, probability_by_edge
        )
        if state_result is None:
            return None
        mass, connected = state_result
        if connected:
            total += mass
    return (total, 2 ** len(edges))


def _math(s, x):
    r = s.get("result", {})
    expected = _expected_result(x)
    if (
        not isinstance(r, dict)
        or set(r) != {"probability", "states"}
        or type(r.get("states")) is not int
        or (expected is None)
    ):
        return False
    probability, states = expected
    return (
        _submission_frac(r.get("probability")) == probability and r["states"] == states
    )


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
