import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W, E = Path("/app"), Path("/tests")


def _is_small_odd_prime(p):
    """Bound p before any exponentiation so a huge probe cannot OOM the verifier."""

    if type(p) is not int or p < 3 or p > 97 or p % 2 == 0:
        return False
    return all(p % d for d in range(3, int(p**0.5) + 1, 2))


def _json_exact_equal(left: object, right: object) -> bool:
    """Compare JSON values with exact scalar types (reject bool==int, float==int)."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _is_valid_prime_formula(s: str) -> bool:
    """The prime formula must describe a power-of-two expression."""

    return "2" in s and "^" in s


def _is_valid_threshold_rule(s: str) -> bool:
    """The threshold rule must describe the global divisibility property."""

    lower = s.casefold()
    return (
        "n" in lower
        and "k" in lower
        and "2" in lower
        and ("divisible" in lower or "divides" in lower)
    )


def _result_ok(result: Any) -> bool:
    """Validate the piecewise construction semantically.

    The public instruction permits any deterministic power-of-two default
    branch and any odd-prime branch whose divisor sum is a perfect square,
    so the verifier checks the mathematical consequences rather than
    matching canonical formula strings.
    """

    if not isinstance(result, dict) or set(result) != {
        "a_1",
        "default_exponent_offset",
        "prime_formula",
        "threshold_rule",
        "probes",
    }:
        return False
    if (
        type(result["a_1"]) is not int
        or result["a_1"] != 1
        or type(result["default_exponent_offset"]) is not int
        or result["default_exponent_offset"] < 0
        or not isinstance(result["prime_formula"], str)
        or not _is_valid_prime_formula(result["prime_formula"])
        or not isinstance(result["threshold_rule"], str)
        or not _is_valid_threshold_rule(result["threshold_rule"])
    ):
        return False
    probes = result["probes"]
    if not isinstance(probes, list) or not 4 <= len(probes) <= 10:
        return False
    primes = []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {
            "prime",
            "a_p",
            "b_p",
            "square_root",
        }:
            return False
        p = probe["prime"]
        if not _is_small_odd_prime(p):
            return False
        if not all(type(probe[k]) is int for k in ("a_p", "b_p", "square_root")):
            return False
        a_p = probe["a_p"]
        b_p = probe["b_p"]
        root = probe["square_root"]
        # b_p = sum_{d|p} d*a_d = 1*a_1 + p*a_p = 1 + p*a_p.
        # b_p must be a positive perfect square with the stated root.
        # The threshold property requires 2^p | a_p for each odd prime p.
        if a_p <= 0 or root <= 0 or b_p != 1 + p * a_p or b_p != root * root:
            return False
        if a_p % (1 << p) != 0:
            return False
        primes.append(p)
    return len(primes) == len(set(primes))


def _frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and json.loads(raw).get("task_id")
            == "jacobian/divisor-sum-square-sequence-repair"
        )
    except (OSError, ValueError):
        return False


def _witness_is_valid(evidence: Any, expected: dict, result: Any) -> bool:
    """Check evidence certificate shape and exact equality, fail closed on recursion."""

    if not evidence or not isinstance(evidence, dict):
        return False
    if not {"schema_version", "task_id", "result"} <= set(evidence):
        return False
    if type(evidence["schema_version"]) is not str or evidence["schema_version"] != "1":
        return False
    if (
        type(evidence["task_id"]) is not str
        or evidence["task_id"] != expected["task_id"]
    ):
        return False
    try:
        return _json_exact_equal(evidence.get("result"), result)
    except RecursionError:
        return False


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    shape_safe = isinstance(submission, dict) and isinstance(
        submission.get("result"), dict
    )
    result = submission.get("result") if shape_safe else None
    math_ok = bool(_result_ok(result) and _frozen_ok())
    evidence = (
        read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/sequence-construction.json",
        )
        if shape_safe
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
        else None
    )
    witness_ok = _witness_is_valid(evidence, expected, result)
    correct = bool(math_ok and witness_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(witness_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
