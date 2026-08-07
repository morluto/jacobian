import json
from functools import cache
from math import gcd
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission_raw,
    read_evidence_json,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_VERTICES = 20


def _integer(value):
    return type(value) is int


def _parse_certificate(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "vertex_count",
        "edges",
        "claims",
    }:
        return None
    n = value.get("vertex_count")
    edges = value.get("edges")
    claims = value.get("claims")
    if (
        value.get("schema_version") != "1"
        or not _integer(n)
        or not 1 <= n <= MAX_VERTICES
        or not isinstance(edges, list)
        or len(edges) > n * (n - 1) // 2
        or not isinstance(claims, dict)
        or set(claims)
        != {
            "connected",
            "independence_number",
            "neighborhood_independence",
            "local_average",
            "hamiltonian_path_exists",
        }
    ):
        return None

    normalized_edges = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(_integer(endpoint) for endpoint in edge)
        ):
            return None
        u, v = edge
        if not 0 <= u < v < n:
            return None
        normalized_edges.append((u, v))
    if normalized_edges != sorted(normalized_edges) or len(normalized_edges) != len(
        set(normalized_edges)
    ):
        return None

    alpha = claims.get("independence_number")
    local = claims.get("neighborhood_independence")
    average = claims.get("local_average")
    if (
        claims.get("connected") is not True
        or claims.get("hamiltonian_path_exists") is not False
        or not _integer(alpha)
        or not 1 <= alpha <= n
        or not isinstance(local, list)
        or len(local) != n
        or not all(_integer(item) and 0 <= item <= n for item in local)
        or not isinstance(average, dict)
        or set(average) != {"numerator", "denominator"}
    ):
        return None
    numerator = average.get("numerator")
    denominator = average.get("denominator")
    if (
        not _integer(numerator)
        or not _integer(denominator)
        or not 0 <= numerator <= MAX_VERTICES * MAX_VERTICES
        or not 1 <= denominator <= n
        or gcd(numerator, denominator) != 1
    ):
        return None
    return n, normalized_edges, claims


def _adjacency(n, edges):
    adjacency = [0] * n
    for u, v in edges:
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u
    return adjacency


def _connected(adjacency):
    seen = 1
    frontier = 1
    while frontier:
        vertex_bit = frontier & -frontier
        frontier -= vertex_bit
        vertex = vertex_bit.bit_length() - 1
        unseen = adjacency[vertex] & ~seen
        seen |= unseen
        frontier |= unseen
    return seen == (1 << len(adjacency)) - 1


def _independence_oracle(adjacency):
    @cache
    def alpha(mask):
        if not mask:
            return 0
        vertex_bit = mask & -mask
        vertex = vertex_bit.bit_length() - 1
        without_vertex = mask ^ vertex_bit
        return max(
            alpha(without_vertex),
            1 + alpha(without_vertex & ~adjacency[vertex]),
        )

    return alpha


def _has_hamiltonian_path(adjacency):
    n = len(adjacency)
    states = [0] * (1 << n)
    for vertex in range(n):
        states[1 << vertex] = 1 << vertex
    for mask in range(1, 1 << n):
        endpoints = states[mask]
        while endpoints:
            vertex_bit = endpoints & -endpoints
            endpoints -= vertex_bit
            vertex = vertex_bit.bit_length() - 1
            available = adjacency[vertex] & ~mask
            while available:
                next_bit = available & -available
                available -= next_bit
                states[mask | next_bit] |= next_bit
    return bool(states[-1])


def _result_matches(result, *, n, alpha, average):
    if not isinstance(result, dict) or set(result) != {
        "certificate_path",
        "vertex_count",
        "independence_number",
        "local_average",
        "hamiltonian_path_exists",
    }:
        return False
    local_average = result.get("local_average")
    if (
        not _integer(result.get("vertex_count"))
        or not _integer(result.get("independence_number"))
        or not isinstance(local_average, dict)
        or set(local_average) != {"numerator", "denominator"}
        or not _integer(local_average.get("numerator"))
        or not _integer(local_average.get("denominator"))
    ):
        return False
    return bool(
        result.get("certificate_path") == "evidence/counterexample.json"
        and result["vertex_count"] == n
        and result["independence_number"] == alpha
        and local_average == average
        and result.get("hamiltonian_path_exists") is False
    )


def _evaluate(certificate, result):
    parsed = _parse_certificate(certificate)
    if parsed is None:
        return False, False
    n, edges, claims = parsed
    adjacency = _adjacency(n, edges)
    alpha_oracle = _independence_oracle(adjacency)
    alpha = alpha_oracle((1 << n) - 1)
    local = [alpha_oracle(mask) for mask in adjacency]
    total = sum(local)
    divisor = gcd(total, n)
    average = {"numerator": total // divisor, "denominator": n // divisor}
    evidence_valid = True
    correct = bool(
        _connected(adjacency)
        and not _has_hamiltonian_path(adjacency)
        and alpha * n <= n + total
        and claims.get("independence_number") == alpha
        and claims.get("neighborhood_independence") == local
        and claims.get("local_average") == average
        and _result_matches(result, n=n, alpha=alpha, average=average)
    )
    return evidence_valid, correct


def main():
    submission = load_submission_raw()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = bool(
        strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=expected["conclusion"],
            allowed_assurances=frozenset({expected["maximum_assurance"]}),
            verification_record="forbidden",
        )
        and data.get("limitations") == expected["required_limitations"]
    )

    evidence = data.get("evidence")
    certificate = None
    if contract and isinstance(evidence, list) and len(evidence) == 1:
        certificate = read_evidence_json(
            evidence[0],
            expected_path=expected["evidence_path"],
        )
    evidence_valid, math_correct = (
        _evaluate(certificate, data.get("result")) if contract else (False, False)
    )
    evidence_valid = bool(contract and evidence_valid)
    math_correct = bool(contract and math_correct)
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    reward = float(
        all(
            (
                math_correct,
                evidence_valid,
                scope_correct,
                assurance_correct,
            )
        )
        and not false_certification
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
