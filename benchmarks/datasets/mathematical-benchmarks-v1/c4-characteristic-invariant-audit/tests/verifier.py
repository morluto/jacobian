import itertools
import json
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
ROLES = {"C4_FREE_ZERO_COUNT", "MULTIPLE_INDUCED_C4", "CHORDED_C4_ZERO_INDUCED"}
_EVIDENCE_READ_BYTES = 64 * 1024
_MAX_EVIDENCE_LINE_BYTES = 1024 * 1024


def _frozen_source():
    """Load the trusted frozen input from /tests without reading workspace bytes."""
    try:
        frozen = E / "input.json"
        if frozen.is_symlink() or not is_regular_bounded_file(frozen, max_bytes=None):
            return {}
        value = json.loads(frozen.read_bytes())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _graph(value, limits):
    if not isinstance(value, dict) or set(value) != {
        "role",
        "vertex_count",
        "edges",
        "induced_c4_count",
        "c4_free_characteristic",
    }:
        return None
    n, edges = value["vertex_count"], value["edges"]
    if (
        type(n) is not int
        or not limits[0] <= n <= limits[1]
        or not isinstance(edges, list)
        or len(edges) > 36
    ):
        return None
    if any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(type(x) is not int for x in edge)
        for edge in edges
    ):
        return None
    pairs = [tuple(edge) for edge in edges]
    if pairs != sorted(set(pairs)) or any(not (0 <= a < b < n) for a, b in pairs):
        return None
    adjacency = [set() for _ in range(n)]
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for u in adjacency[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    if len(seen) != n:
        return None
    return n, adjacency


def _has_cycle_on(vertices, adjacency):
    first = min(vertices)
    rest = [v for v in vertices if v != first]
    return any(
        all(order[(i + 1) % 4] in adjacency[order[i]] for i in range(4))
        for tail in itertools.permutations(rest)
        for order in [(first, *tail)]
    )


def _invariants(n, adjacency):
    induced = 0
    has_c4 = False
    for vertices in itertools.combinations(range(n), 4):
        cycle = _has_cycle_on(vertices, adjacency)
        has_c4 |= cycle
        edge_count = sum(
            v in adjacency[u] for u, v in itertools.combinations(vertices, 2)
        )
        degrees = [sum(v in adjacency[u] for v in vertices if v != u) for u in vertices]
        induced += int(edge_count == 4 and degrees == [2, 2, 2, 2])
    return induced, int(not has_c4)


def _is_exact_int(value):
    """Reject JSON booleans that compare equal to 0 or 1."""
    return type(value) is int


def _witness_valid(witness, contracts):
    """Validate a single graph witness, rejecting booleans and bad roles."""
    role = witness.get("role")
    if not _is_exact_int(witness.get("induced_c4_count")):
        return False
    if not _is_exact_int(witness.get("c4_free_characteristic")):
        return False
    parsed = _graph(witness, contracts.get(role, (-1, -1)))
    if parsed is None:
        return False
    induced, characteristic = _invariants(*parsed)
    if (
        witness["induced_c4_count"] != induced
        or witness["c4_free_characteristic"] != characteristic
    ):
        return False
    if role == "C4_FREE_ZERO_COUNT" and not (induced == 0 and characteristic == 1):
        return False
    if role == "MULTIPLE_INDUCED_C4" and not (induced >= 2 and characteristic == 0):
        return False
    if role == "CHORDED_C4_ZERO_INDUCED":
        return induced == 0 and characteristic == 0
    return True


def _valid(result, source):
    if (
        not isinstance(result, dict)
        or set(result) != {"semantic_status", "witnesses"}
        or result.get("semantic_status") != "NOT_EQUIVALENT"
    ):
        return False
    witnesses = result.get("witnesses")
    contracts = {
        x["role"]: (x["min_vertices"], x["max_vertices"])
        for x in source.get("witness_contracts", [])
        if isinstance(x, dict)
    }
    if (
        not isinstance(witnesses, list)
        or len(witnesses) != 3
        or not all(isinstance(x, dict) for x in witnesses)
    ):
        return False
    # Validate roles are hashable strings before constructing the set.
    if not all(isinstance(x.get("role"), str) for x in witnesses):
        return False
    if {x.get("role") for x in witnesses} != ROLES:
        return False
    return all(_witness_valid(witness, contracts) for witness in witnesses)


def _raw_submission():
    """Parse the bounded submission without applying the public schema."""
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    submission, source = load_submission(require_input_binding=False), _frozen_source()
    input_bound = workspace_input_is_bound(W / "input.json", tests=E)
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(_valid(result, source))
    correct = bool(math_correct and input_bound)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
