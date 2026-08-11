import json
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _load_bound_input() -> dict[str, Any]:
    try:
        visible = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if visible.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if visible.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _adjacency(
    vertices: list[str], edges: list[list[str]]
) -> dict[str, set[str]] | None:
    if len(vertices) != len(set(vertices)):
        return None
    adjacency = {vertex: set() for vertex in vertices}
    seen: set[frozenset[str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in adjacency
            or edge[1] not in adjacency
            or edge[0] == edge[1]
        ):
            return None
        key = frozenset(edge)
        if key in seen:
            return None
        seen.add(key)
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency


def _connected(adjacency: dict[str, set[str]]) -> bool:
    if not adjacency:
        return False
    first = next(iter(adjacency))
    reached = {first}
    queue = deque([first])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return reached == set(adjacency)


def _cover_data(
    source: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]] | None:
    try:
        base = source["base_graph"]
        cover = source["cover_graph"]
        mapping = source["covering_map"]
        base_adj = _adjacency(base["vertices"], base["edges"])
        cover_adj = _adjacency(cover["vertices"], cover["edges"])
    except (KeyError, TypeError):
        return None
    if (
        base_adj is None
        or cover_adj is None
        or not isinstance(mapping, dict)
        or set(mapping) != set(cover_adj)
        or any(image not in base_adj for image in mapping.values())
        or not _connected(base_adj)
        or not _connected(cover_adj)
    ):
        return None
    for vertex, neighbors in cover_adj.items():
        images = [mapping[neighbor] for neighbor in neighbors]
        if len(images) != len(set(images)) or set(images) != base_adj[mapping[vertex]]:
            return None
    return base_adj, cover_adj, mapping


def _path_is_valid(
    path: object, source: dict[str, Any], base_adj: dict[str, set[str]]
) -> bool:
    if not isinstance(path, list) or not all(isinstance(v, str) for v in path):
        return False
    contract = source["path_contract"]
    endpoints = source["endpoints"]
    edges = len(path) - 1
    return bool(
        path
        and path[0] == endpoints["source"]
        and path[-1] == endpoints["target"]
        and contract["minimum_edges"] <= edges <= contract["maximum_edges"]
        and (not contract["simple"] or len(path) == len(set(path)))
        and all(right in base_adj[left] for left, right in pairwise(path))
    )


def _unique_lift(
    path: list[str], start: str, cover_adj: dict[str, set[str]], mapping: dict[str, str]
) -> list[str] | None:
    if mapping.get(start) != path[0]:
        return None
    trace = [start]
    current = start
    for image in path[1:]:
        candidates = sorted(v for v in cover_adj[current] if mapping[v] == image)
        if len(candidates) != 1:
            return None
        current = candidates[0]
        trace.append(current)
    return trace


def _submitted_lifts(value: object) -> dict[str, tuple[list[str], str]] | None:
    if not isinstance(value, list):
        return None
    result: dict[str, tuple[list[str], str]] = {}
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"start", "trace", "endpoint"}
            or not isinstance(item["start"], str)
            or not isinstance(item["trace"], list)
            or not all(isinstance(v, str) for v in item["trace"])
            or not isinstance(item["endpoint"], str)
            or item["start"] in result
        ):
            return None
        result[item["start"]] = (item["trace"], item["endpoint"])
    return result


def _lifts_match(
    submitted: object,
    path: list[str],
    starts: set[str],
    cover_adj: dict[str, set[str]],
    mapping: dict[str, str],
) -> dict[str, str] | None:
    lifts = _submitted_lifts(submitted)
    if lifts is None or set(lifts) != starts:
        return None
    endpoints: dict[str, str] = {}
    for start in starts:
        expected = _unique_lift(path, start, cover_adj, mapping)
        trace, endpoint = lifts[start]
        if expected is None or trace != expected or endpoint != expected[-1]:
            return None
        endpoints[start] = endpoint
    return endpoints


def _result_is_valid(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "base_path",
        "reverse_path",
        "forward_lifts",
        "reverse_lifts",
        "bijection",
        "local_covering_checked",
    }:
        return False
    data = _cover_data(source)
    if data is None or value["local_covering_checked"] is not True:
        return False
    base_adj, cover_adj, mapping = data
    path = value["base_path"]
    if not _path_is_valid(path, source, base_adj) or value["reverse_path"] != list(
        reversed(path)
    ):
        return False
    source_fiber = {
        v for v, image in mapping.items() if image == source["endpoints"]["source"]
    }
    target_fiber = {
        v for v, image in mapping.items() if image == source["endpoints"]["target"]
    }
    forward = _lifts_match(
        value["forward_lifts"], path, source_fiber, cover_adj, mapping
    )
    if forward is None or set(forward.values()) != target_fiber:
        return False
    reverse = _lifts_match(
        value["reverse_lifts"], list(reversed(path)), target_fiber, cover_adj, mapping
    )
    if reverse is None or any(
        reverse[target] != start for start, target in forward.items()
    ):
        return False
    bijection = value["bijection"]
    if not isinstance(bijection, list) or len(bijection) != len(source_fiber):
        return False
    if any(
        not isinstance(item, dict)
        or set(item) != {"source", "target"}
        or not isinstance(item["source"], str)
        or not isinstance(item["target"], str)
        for item in bijection
    ):
        return False
    pairs = {(item["source"], item["target"]) for item in bijection}
    return len(pairs) == len(source_fiber) and pairs == set(forward.items())


def _evidence_is_valid(value: object, result: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    target = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().lower()
    except (OSError, UnicodeError):
        return False
    if not isinstance(result, dict) or not isinstance(result.get("bijection"), list):
        return False
    return bool(
        all(
            term in text for term in ("unique", "reversed path", "bijection", "inverse")
        )
        and not any(
            marker in text for marker in ("not a bijection", "not unique", "no inverse")
        )
        and all(
            str(pair.get(key, "")).casefold() in text
            for pair in result["bijection"]
            if isinstance(pair, dict)
            for key in ("source", "target")
        )
    )


def main() -> None:
    submission = load_submission()
    expected = json.loads((TESTS / "expected.json").read_text())
    source = _load_bound_input()
    math_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="optional",
    )
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    source_ok = (
        source.get("source", {}).get("row_sha256")
        == "sha256:a0b2d9c381ca5ac7596b5940810c19b09be1cbe8f3cc1d123e0eba398f2a00c2"
    )
    math_correct = bool(
        math_contract
        and source_ok
        and submission.get("completeness") == "COMPLETE"
        and _result_is_valid(submission.get("result"), source)
    )
    evidence = bool(
        math_contract
        and _evidence_is_valid(submission.get("evidence"), submission.get("result"))
    )
    scope = bool(
        math_contract
        and isinstance(submission.get("scope"), str)
        and all(
            term in submission["scope"].casefold()
            for term in ("finite", "graph", "cover", "fiber", "path")
        )
        and not any(
            marker in submission["scope"].casefold()
            for marker in ("not ", "without", "cannot", "does not")
        )
    )
    assurance = bool(
        math_contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    correct = bool(contract and math_correct and evidence and scope and assurance)
    reward = (
        0.0
        if not correct or false_certification
        else 0.8 + 0.1 * scope + 0.1 * assurance
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(output)


if __name__ == "__main__":
    main()
