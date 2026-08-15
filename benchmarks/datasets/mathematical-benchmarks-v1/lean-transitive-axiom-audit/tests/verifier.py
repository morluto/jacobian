import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The checker does not execute Lean, assess proposition truth, or establish the behavior of current Lean releases."


def _load_input() -> dict[str, Any]:
    try:
        visible, hidden = WORKSPACE / "input.json", TESTS / "input.json"
        if (
            visible.is_symlink()
            or hidden.is_symlink()
            or visible.read_bytes() != hidden.read_bytes()
        ):
            return {}
        value = json.loads(hidden.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _closure(roots: list[str], deps: dict[str, list[str]]) -> list[str] | None:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        if node not in deps or not isinstance(deps[node], list):
            return None
        seen.add(node)
        pending.extend(deps[node])
    return sorted(seen)


def _expected(case: object) -> dict[str, Any] | None:
    if not isinstance(case, dict):
        return None
    roots, observed, deps, roles = (
        case.get("roots"),
        case.get("observed"),
        case.get("dependencies"),
        case.get("roles"),
    )
    if (
        not isinstance(roots, list)
        or not isinstance(observed, list)
        or not isinstance(deps, dict)
        or not isinstance(roles, dict)
    ):
        return None
    if any(not isinstance(x, str) for x in roots + observed):
        return None
    if any(
        not isinstance(k, str)
        or not isinstance(v, list)
        or any(not isinstance(x, str) for x in v)
        for k, v in deps.items()
    ):
        return None
    closure = _closure(roots, deps)
    if closure is None:
        return None
    missing = sorted(set(closure) - set(observed))
    if any(name not in roles for name in missing):
        return None
    return {
        "case_id": case.get("case_id"),
        "status": "INCOMPLETE" if missing else "COMPLETE",
        "closure": closure,
        "missing": missing,
        "missing_roles": {name: roles[name] for name in missing},
    }


def _result_valid(result: object, frozen: dict[str, Any]) -> bool:
    cases = frozen.get("cases")
    if (
        not isinstance(result, dict)
        or set(result) != {"cases"}
        or not isinstance(cases, list)
        or not isinstance(result["cases"], list)
    ):
        return False
    expected = [_expected(case) for case in cases]
    if any(item is None for item in expected):
        return False
    submitted = {
        item.get("case_id"): item for item in result["cases"] if isinstance(item, dict)
    }
    return len(submitted) == len(result["cases"]) == len(expected) and all(
        submitted.get(item["case_id"]) == item for item in expected if item
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(submission and _result_valid(data.get("result"), _load_input()))
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(math_correct),
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
