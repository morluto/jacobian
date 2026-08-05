import json
from fractions import Fraction
from itertools import product
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
ALLOWED = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})


def _frac(v):
    if not isinstance(v, dict) or set(v) != {"num", "den"}:
        return None
    num, den = v.get("num"), v.get("den")
    if (
        not isinstance(num, str)
        or not isinstance(den, str)
        or not num.lstrip("-").isdigit()
        or not den.isdigit()
    ):
        return None
    try:
        value = Fraction(int(num), int(den))
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return value


def _graph_parts(x):
    graph = x.get("graph")
    probabilities = x.get("edge_probabilities")
    terminals = x.get("terminals")
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("vertices"), list)
        or not isinstance(graph.get("edges"), list)
        or not isinstance(probabilities, list)
        or not isinstance(terminals, list)
        or len(terminals) != 2
    ):
        return None
    return graph["vertices"], graph["edges"], probabilities, terminals


def _probabilities_by_edge(probabilities, edges):
    probability_by_edge = {}
    for item in probabilities:
        if not isinstance(item, dict) or set(item) != {"edge", "open_probability"}:
            return None
        edge = item["edge"]
        probability = _frac(item["open_probability"])
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(vertex, str) for vertex in edge)
            or probability is None
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
    return mass, terminals[1] in reached


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
    return total, 2 ** len(edges)


def _math(s, x, e):
    r = s.get("result", {})
    expected = _expected_result(x)
    if (
        not isinstance(r, dict)
        or set(r) != {"probability", "states"}
        or type(r.get("states")) is not int
        or expected is None
    ):
        return False
    probability, states = expected
    return _frac(r.get("probability")) == probability and r["states"] == states


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
