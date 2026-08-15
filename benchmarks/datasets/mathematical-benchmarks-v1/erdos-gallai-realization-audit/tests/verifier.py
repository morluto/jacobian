import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The checker does not enumerate all graph realizations or prove claims about other degree sequences."


def _load() -> dict[str, Any]:
    try:
        a, b = WORKSPACE / "input.json", TESTS / "input.json"
        if a.is_symlink() or b.is_symlink() or a.read_bytes() != b.read_bytes():
            return {}
        x = json.loads(b.read_text())
    except (OSError, ValueError):
        return {}
    return x if isinstance(x, dict) else {}


def _violations(ds: list[int]) -> list[dict[str, int]]:
    out = []
    for k in range(1, len(ds) + 1):
        lhs = sum(ds[:k])
        rhs = k * (k - 1) + sum(min(d, k) for d in ds[k:])
        if lhs > rhs:
            out.append({"k": k, "lhs": lhs, "rhs": rhs})
    return out


def _edges_valid(edges: object, ds: list[int]) -> bool:
    if not isinstance(edges, list):
        return False
    raw_edges = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(v) is not int for v in edge)
        ):
            return False
        raw_edges.append(tuple(edge))
    for offset in (0, 1):
        normalized = []
        for raw_u, raw_v in raw_edges:
            u, v = raw_u - offset, raw_v - offset
            if not 0 <= u < len(ds) or not 0 <= v < len(ds) or u == v:
                break
            normalized.append(tuple(sorted((u, v))))
        else:
            if len(normalized) != len(set(normalized)):
                continue
            degrees = [
                sum(vertex in edge for edge in normalized) for vertex in range(len(ds))
            ]
            if degrees == ds:
                return True
    return False


def _expected(case: dict[str, Any], submitted: dict[str, Any]) -> bool:
    ds = case.get("degrees")
    if (
        not isinstance(ds, list)
        or any(type(x) is not int for x in ds)
        or ds != sorted(ds, reverse=True)
    ):
        return False
    violations = _violations(ds)
    parity = sum(ds) % 2
    graphical = not violations and parity == 0
    if submitted.get("case_id") != case.get("case_id"):
        return False
    if graphical:
        return (
            submitted.get("status") == "GRAPHICAL"
            and submitted.get("violations") == []
            and _edges_valid(submitted.get("edges"), ds)
        )
    return (
        submitted.get("status") == "NONGRAPHICAL"
        and submitted.get("edges") == []
        and submitted.get("violations") == violations
    )


def _result(result: object, frozen: dict[str, Any]) -> bool:
    source = frozen.get("cases")
    if (
        not isinstance(result, dict)
        or set(result) != {"cases"}
        or not isinstance(source, list)
        or not isinstance(result["cases"], list)
    ):
        return False
    if any(
        not isinstance(item, dict)
        or set(item) != {"case_id", "status", "edges", "violations"}
        or not isinstance(item.get("case_id"), str)
        for item in result["cases"]
    ):
        return False
    by_id = {x["case_id"]: x for x in result["cases"]}
    return len(by_id) == len(result["cases"]) == len(source) and all(
        isinstance(c, dict)
        and isinstance(by_id.get(c.get("case_id")), dict)
        and _expected(c, by_id[c["case_id"]])
        for c in source
    )


def main() -> None:
    s = load_submission()
    d = s if isinstance(s, dict) else {}
    math_correct = bool(_result(d.get("result"), _load()))
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(math_correct),
            }
        )
    )
    normalize_reward_file(p / "reward.json")


if __name__ == "__main__":
    main()
