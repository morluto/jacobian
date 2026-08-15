from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/bounded-variation-uniform-limit"
CONCLUSION = "UNIFORM_CONVERGENCE_DOES_NOT_FORCE_VARIATION_CONVERGENCE"
SCOPE = "the full sequence on [0,2*pi] and all submitted exact checkpoints"
LIMITATION = "NO_PROOF_ASSISTANT_VERIFICATION"

# Accept any ordering of q, n, x in the sine argument and q, n in the
# denominator, with or without explicit multiplication signs.
_SEQUENCE_RE = re.compile(r"^sin\(([qnx])\*([qnx])\*([qnx])\)/\(([qn])\*([qn])\)$")


def _is_int(value: object) -> bool:
    """Accept only true integers, rejecting booleans and floats."""

    return type(value) is int


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value
    ):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _valid_sequence(seq: object) -> bool:
    """Accept equivalent serializations of ``sin(q*n*x)/(q*n)``."""

    if not isinstance(seq, str):
        return False
    match = _SEQUENCE_RE.fullmatch(seq.replace(" ", ""))
    return bool(
        match
        and set(match.group(1, 2, 3)) == {"q", "n", "x"}
        and set(match.group(4, 5)) == {"q", "n"}
    )


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        workspace_input = WORKSPACE / "input.json"
        return (
            is_regular_bounded_file(workspace_input, max_bytes=16 * 1024 * 1024)
            and workspace_input.read_bytes() == hidden
            and data["source"]["row"] == 600
            and data["source"]["revision"] == "d4e9f8ca877552f4491a9c2d52e0d230c0fca620"
        )
    except (OSError, ValueError, KeyError, MemoryError):
        return False


def _argument_ok(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"uniform_convergence", "variation_behavior", "implication"}
        and value["uniform_convergence"] == "SUP_NORM_1_OVER_QN_TENDS_TO_ZERO"
        and value["variation_behavior"] == "TOTAL_VARIATION_IS_CONSTANTLY_FOUR"
        and value["implication"] == CONCLUSION
    )


def _uniform_certificate_ok(value: object, q: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {"sup_norm_numerator", "sup_norm_denominator_coefficient", "tends_to_zero"}
        and _is_int(value.get("sup_norm_numerator"))
        and value.get("sup_norm_numerator") == 1
        and _is_int(value.get("sup_norm_denominator_coefficient"))
        and value.get("sup_norm_denominator_coefficient") == q
        and value.get("tends_to_zero") is True
    )


def _variation_formula_ok(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "endpoint_segment_count",
            "interior_segment_count",
            "endpoint_jump_multiplier",
            "interior_jump_multiplier",
            "total_variation",
        }
        and _is_int(value.get("endpoint_segment_count"))
        and value.get("endpoint_segment_count") == 2
        and isinstance(value.get("interior_segment_count"), str)
        and value.get("interior_segment_count") == "2*q*n-1"
        and _is_int(value.get("endpoint_jump_multiplier"))
        and value.get("endpoint_jump_multiplier") == 1
        and _is_int(value.get("interior_jump_multiplier"))
        and value.get("interior_jump_multiplier") == 2
        and isinstance(value.get("total_variation"), str)
        and value.get("total_variation") == "4"
    )


def _checkpoint_ok(item: object, q: int, seen: set[int]) -> bool:
    if not isinstance(item, dict) or set(item) != {
        "n",
        "frequency",
        "amplitude",
        "interior_segments",
        "endpoint_contribution",
        "interior_contribution",
        "total_variation",
    }:
        return False
    n = item["n"]
    if not _is_int(n) or n < 1 or n in seen:
        return False
    seen.add(n)
    frequency = q * n
    amplitude = Fraction(1, frequency)
    interior_segments = 2 * frequency - 1
    endpoint = 2 * amplitude
    interior = interior_segments * 2 * amplitude
    return (
        _is_int(item.get("frequency"))
        and item.get("frequency") == frequency
        and _is_int(item.get("interior_segments"))
        and item.get("interior_segments") == interior_segments
        and _fraction(item["amplitude"]) == amplitude
        and _fraction(item["endpoint_contribution"]) == endpoint
        and _fraction(item["interior_contribution"]) == interior
        and _fraction(item["total_variation"]) == 4
        and endpoint + interior == 4
    )


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "scale_q",
        "sequence",
        "limit_function",
        "argument",
        "uniform_certificate",
        "variation_formula",
        "checkpoints",
    }:
        return False
    q = value["scale_q"]
    if not _is_int(q) or not 2 <= q <= 9:
        return False
    if not _valid_sequence(value["sequence"]) or value["limit_function"] != "0":
        return False
    if not _argument_ok(value["argument"]):
        return False
    if not _uniform_certificate_ok(value["uniform_certificate"], q):
        return False
    if not _variation_formula_ok(value["variation_formula"]):
        return False
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list) or not 4 <= len(checkpoints) <= 10:
        return False
    seen: set[int] = set()
    return all(_checkpoint_ok(item, q, seen) for item in checkpoints)


def _evaluate(submission: object) -> dict[str, float | bool]:
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(_source_is_bound() and _result(data.get("result")))
    reward = float(math_correct)
    return {
        "correctness": float(math_correct),
        "reward": reward,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
