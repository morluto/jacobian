from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    _finite_json_float,
    _reject_duplicate_keys,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/totient-preimage-completeness-certificate"
SCOPE = "phi-48-complete-preimage-classification-v1"
LIMITATIONS = [
    "ONE_TARGET_TOTIENT_VALUE_48",
    "EXACT_FINITE_PREIMAGE_CLASSIFICATION",
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


def _candidate_certificate_valid(
    result: dict[str, Any], primes: list[int], options: list[list[int]]
) -> bool:
    if "candidate_primes" in result:
        candidate_primes = result["candidate_primes"]
        if (
            not isinstance(candidate_primes, list)
            or len(candidate_primes) != len(primes)
            or any(type(prime) is not int for prime in candidate_primes)
            or set(candidate_primes) != set(primes)
        ):
            return False
    if "prime_power_options" in result:
        submitted_options = result["prime_power_options"]
        if not isinstance(submitted_options, list) or len(submitted_options) != len(
            primes
        ):
            return False
        observed_options: dict[int, list[int]] = {}
        for option in submitted_options:
            if (
                not isinstance(option, dict)
                or set(option) != {"prime", "exponents"}
                or type(option["prime"]) is not int
                or not isinstance(option["exponents"], list)
                or any(type(exponent) is not int for exponent in option["exponents"])
                or option["prime"] in observed_options
            ):
                return False
            observed_options[option["prime"]] = option["exponents"]
        if observed_options != dict(zip(primes, options, strict=True)):
            return False
    return "enumerated_branch_count" not in result or (
        type(result["enumerated_branch_count"]) is int
        and result["enumerated_branch_count"] == math.prod(map(len, options))
    )


def _solutions_valid(
    rows: Any, expected: dict[int, list[list[int]]]
) -> dict[int, list[list[int]]] | None:
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    observed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"n", "factorization", "totient"}:
            return None
        n, factors, totient = row["n"], row["factorization"], row["totient"]
        if (
            type(n) is not int
            or type(totient) is not int
            or not isinstance(factors, list)
        ):
            return None
        if n not in expected:
            return None
        normalized_factors: dict[int, int] = {}
        for factor in factors:
            if (
                not isinstance(factor, list)
                or len(factor) != 2
                or any(type(value) is not int for value in factor)
                or factor[0] in normalized_factors
                or factor[0] < 2
                or factor[1] < 1
            ):
                return None
            normalized_factors[factor[0]] = factor[1]
        expected_factors = dict(expected.get(n, []))
        if (
            n in observed
            or normalized_factors != expected_factors
            or math.prod(p**a for p, a in normalized_factors.items()) != n
            or totient != 48
        ):
            return None
        observed[n] = expected[n]
    return observed


def mathematics(result: Any) -> bool:
    required = {"solutions", "accepted_count"}
    if not isinstance(result, dict) or not required.issubset(result):
        return False
    primes = _candidate_primes()
    options = [_options(p) for p in primes]
    if not _candidate_certificate_valid(result, primes, options):
        return False
    expected = _expected_solutions(primes, options)
    observed = _solutions_valid(result["solutions"], expected)
    return (
        observed == expected
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
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
            parse_float=_finite_json_float,
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
            raw["evidence"][0],
            expected_path="evidence/answer.json",
            max_bytes=None,
        )
        if evidence_ok
        else None
    )
    try:
        evidence_ok = bool(
            isinstance(payload, dict)
            and set(payload) == {"schema_version", "task_id", "result", "limitations"}
            and payload.get("schema_version") == "1"
            and type(raw.get("task_id")) is str
            and _json_equal(payload.get("task_id"), raw["task_id"])
            and _json_equal(payload.get("result"), raw.get("result"))
            and _json_equal(payload.get("limitations"), raw.get("limitations"))
        )
    except RecursionError:
        evidence_ok = False
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("completeness") == "COMPLETE"
        and raw.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        isinstance(raw, dict)
        and type(raw.get("claimed_assurance")) is str
        and raw.get("claimed_assurance") in {"UNVERIFIED", "COMPUTED", "CHECKED"}
    )
    values = {
        "input_binding": float(bound),
        "protocol": float(bool(contract)),
        "correctness": float(math_ok),
        "mathematics": float(math_ok),
        "evidence": float(evidence_ok),
        "evidence_validity": float(evidence_ok),
        "scope": float(scope_ok),
        "scope_accuracy": float(scope_ok),
        "assurance": float(assurance_ok),
        "assurance_calibration": float(assurance_ok),
    }
    values["aggregate_reward"] = float(
        all(
            values[key]
            for key in (
                "input_binding",
                "protocol",
                "correctness",
                "evidence_validity",
                "scope_accuracy",
                "assurance_calibration",
            )
        )
    )
    values["reward"] = values["aggregate_reward"]
    values["false_certification"] = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
    )
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    reward_path = path / "reward.json"
    reward_path.write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(reward_path)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        path = Path("/logs/verifier")
        path.mkdir(parents=True, exist_ok=True)
        reward_path = path / "reward.json"
        reward_path.write_text(
            json.dumps(
                {
                    "protocol": 0.0,
                    "input_binding": 0.0,
                    "correctness": 0.0,
                    "mathematics": 0.0,
                    "evidence": 0.0,
                    "evidence_validity": 0.0,
                    "scope": 0.0,
                    "scope_accuracy": 0.0,
                    "assurance": 0.0,
                    "assurance_calibration": 0.0,
                    "false_certification": False,
                    "aggregate_reward": 0.0,
                    "reward": 0.0,
                    "error": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        normalize_reward_file(reward_path)
