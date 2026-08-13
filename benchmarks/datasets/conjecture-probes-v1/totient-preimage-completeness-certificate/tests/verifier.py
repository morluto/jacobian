from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/totient-preimage-completeness-certificate"
SCOPE = "phi-48-complete-preimage-classification-v1"
LIMITATIONS = [
    "ONE_TARGET_TOTIENT_VALUE_48",
    "EXACT_PRIME_POWER_BRANCH_ENUMERATION",
    "NO_GLOBAL_CARMICHAEL_CONCLUSION",
]


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _candidate_primes() -> list[int]:
    return [p for p in range(2, 50) if _prime(p) and 48 % (p - 1) == 0]


def _contribution(p: int, exponent: int) -> int:
    return 1 if exponent == 0 else (p - 1) * p ** (exponent - 1)


def _options(p: int) -> list[int]:
    values = [0]
    exponent = 1
    while 48 % _contribution(p, exponent) == 0:
        values.append(exponent)
        exponent += 1
    return values


def _expected_solutions(
    primes: list[int], options: list[list[int]]
) -> dict[int, list[list[int]]]:
    result = {}
    for exponents in itertools.product(*options):
        if (
            math.prod(
                _contribution(p, a) for p, a in zip(primes, exponents, strict=True)
            )
            != 48
        ):
            continue
        factors = [[p, a] for p, a in zip(primes, exponents, strict=True) if a]
        result[math.prod(p**a for p, a in factors)] = factors
    return result


def mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "candidate_primes",
        "prime_power_options",
        "enumerated_branch_count",
        "solutions",
        "accepted_count",
    }:
        return False
    primes = _candidate_primes()
    options = [_options(p) for p in primes]
    if result["candidate_primes"] != primes:
        return False
    expected_options = [
        {"prime": p, "exponents": values}
        for p, values in zip(primes, options, strict=True)
    ]
    if result["prime_power_options"] != expected_options:
        return False
    expected = _expected_solutions(primes, options)
    rows = result["solutions"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    observed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"n", "factorization", "totient"}:
            return False
        n, factors, totient = row["n"], row["factorization"], row["totient"]
        if (
            type(n) is not int
            or type(totient) is not int
            or not isinstance(factors, list)
        ):
            return False
        if n in observed or expected.get(n) != factors or totient != 48:
            return False
        observed[n] = factors
    return (
        observed == expected
        and type(result["enumerated_branch_count"]) is int
        and result["enumerated_branch_count"] == math.prod(map(len, options))
        and type(result["accepted_count"]) is int
        and result["accepted_count"] == len(expected)
    )


def _json_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _raw() -> dict[str, Any] | None:
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(
            path.read_text(),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="PHI_48_COMPLETE_PREIMAGE_CLASSIFICATION",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    raw = _raw()
    math_ok = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.txt", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and _json_equal(payload.get("result"), raw.get("result"))
        and payload.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("completeness") == "COMPLETE"
        and raw.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        contract
        and isinstance(raw, dict)
        and raw.get("claimed_assurance") in {"UNVERIFIED", "COMPUTED", "CHECKED"}
    )
    values = {
        "input_binding": float(bound),
        "protocol": float(bool(contract)),
        "mathematics": float(math_ok),
        "evidence": float(evidence_ok),
        "scope": float(scope_ok),
        "assurance": float(assurance_ok),
    }
    values["aggregate_reward"] = float(all(values.values()))
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
