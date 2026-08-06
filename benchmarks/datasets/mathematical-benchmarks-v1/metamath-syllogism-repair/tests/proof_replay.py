from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifier_support import resolve_evidence, strict_submission_contract

VARIABLES = {"u", "v", "w"}

ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
LIMITATIONS = [
    "FROZEN_FRAGMENT_NOT_FULL_UPSTREAM_DATABASE",
    "NO_EXTERNAL_METAMATH_KERNEL_REPLAY",
]
RESULT_KEYS = frozenset(
    {"repaired_proof", "changed_positions", "trace", "final_expression"}
)
STEP_KEYS = frozenset({"position", "label", "substitution", "stack_depth", "stack_top"})


@dataclass
class VerifyResult:
    """Independent diagnostic dimensions for one submission."""

    protocol_ok: bool
    correctness: bool
    evidence_validity: bool
    scope_accuracy: bool
    assurance_ok: bool
    message: str


def _capture_expression(actual: list[str], start: int) -> tuple[list[str], int]:
    if start >= len(actual):
        raise ValueError("missing variable expression")
    if actual[start] != "(":
        return [actual[start]], start + 1
    depth = 0
    for end in range(start, len(actual)):
        depth += actual[end] == "("
        depth -= actual[end] == ")"
        if depth == 0:
            return actual[start : end + 1], end + 1
    raise ValueError("unbalanced expression")


def _unify_variable(
    token: str,
    actual: list[str],
    start: int,
    substitution: dict[str, list[str]],
) -> int:
    value = substitution.get(token)
    if value is None:
        value, _ = _capture_expression(actual, start)
        substitution[token] = value
    if actual[start : start + len(value)] != value:
        raise ValueError("inconsistent substitution")
    return start + len(value)


def _unify(
    pattern: list[str], actual: list[str], substitution: dict[str, list[str]]
) -> None:
    actual_index = 0
    for token in pattern:
        if token in VARIABLES:
            actual_index = _unify_variable(token, actual, actual_index, substitution)
        else:
            if actual_index >= len(actual) or actual[actual_index] != token:
                raise ValueError("token mismatch")
            actual_index += 1
    if actual_index != len(actual):
        raise ValueError("unconsumed expression tokens")


def _instantiate(
    expression: list[str], substitution: dict[str, list[str]]
) -> list[str]:
    output: list[str] = []
    for token in expression:
        output.extend(substitution.get(token, [token]))
    return output


def replay(input_data: dict[str, Any], proof: list[str]) -> list[dict[str, Any]]:
    stack: list[list[str]] = []
    trace: list[dict[str, Any]] = []
    atomic = input_data["atomic_entries"]
    assertions = input_data["assertions"]
    for position, label in enumerate(proof):
        substitution: dict[str, list[str]] = {}
        if label in atomic:
            stack.append(list(atomic[label]))
        elif label in assertions:
            assertion = assertions[label]
            hypotheses = assertion["hypotheses"]
            if len(stack) < len(hypotheses):
                raise ValueError("stack underflow")
            actuals = stack[-len(hypotheses) :]
            del stack[-len(hypotheses) :]
            for pattern, actual in zip(hypotheses, actuals, strict=True):
                _unify(pattern, actual, substitution)
            stack.append(_instantiate(assertion["conclusion"], substitution))
        else:
            raise ValueError("unknown label")
        trace.append(
            {
                "position": position,
                "label": label,
                "substitution": substitution,
                "stack_depth": len(stack),
                "stack_top": stack[-1],
            }
        )
    if stack != [input_data["target"]]:
        raise ValueError("proof does not finish at the target")
    return trace


def _is_int(value: object) -> bool:
    """Reject booleans where integers are required."""

    return type(value) is int


def _trace_types_valid(trace: object) -> bool:
    """Validate exact step shapes and integer field types before comparison."""

    if not isinstance(trace, list) or len(trace) != 11:
        return False
    for step in trace:
        if not isinstance(step, dict) or set(step) != STEP_KEYS:
            return False
        if not _is_int(step["position"]) or not _is_int(step["stack_depth"]):
            return False
        if not isinstance(step["label"], str):
            return False
        if not isinstance(step["substitution"], dict):
            return False
        if not isinstance(step["stack_top"], list):
            return False
    return True


def _evidence_binds_result(
    evidence: object, result: dict[str, Any], task_root: Path
) -> bool:
    """Bind evidence to the exact path, digest, result, and replay explanation."""

    target = resolve_evidence(
        evidence[0] if isinstance(evidence, list) and len(evidence) == 1 else None,
        expected_path="evidence/answer.txt",
        workspace=task_root,
    )
    if target is None:
        return False
    try:
        lines = target.read_text().splitlines()
    except (OSError, UnicodeError, MemoryError):
        return False
    marker_lines = [line for line in lines if line.startswith("RESULT_JSON:")]
    if len(marker_lines) != 1:
        return False
    try:
        marker_value = json.loads(marker_lines[0].removeprefix("RESULT_JSON:").strip())
    except (TypeError, ValueError, RecursionError, MemoryError):
        return False
    if json.dumps(marker_value, sort_keys=True, separators=(",", ":")) != json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ):
        return False
    explanation = " ".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("RESULT_JSON:")
    ).casefold()
    return bool(
        explanation
        and "ordered" in explanation
        and "hypothes" in explanation
        and "stack" in explanation
        and "unif" in explanation
        and "substitut" in explanation
        and all(label in explanation for label in ("wi", "a1i", "mpd"))
    )


def _valid_proof_and_positions(
    result: dict[str, Any], input_data: dict[str, Any]
) -> tuple[list[str] | None, str | None]:
    proof = result.get("repaired_proof")
    if not isinstance(proof, list) or not all(type(label) is str for label in proof):
        return None, "missing proof"
    corrupted = input_data.get("corrupted_proof")
    if not isinstance(corrupted, list) or len(proof) != len(corrupted):
        return None, "proof length mismatch"
    changed = [
        index
        for index, (before, after) in enumerate(zip(corrupted, proof, strict=True))
        if before != after
    ]
    submitted_positions = result.get("changed_positions")
    if not isinstance(submitted_positions, list) or not all(
        _is_int(position) for position in submitted_positions
    ):
        return None, "changed positions mismatch"
    if sorted(changed) != sorted(submitted_positions):
        return None, "changed positions mismatch"
    required = input_data.get("required_replacements")
    if not _is_int(required) or len(changed) != required:
        return None, "wrong replacement count"
    return proof, None


def _valid_trace_and_target(
    result: dict[str, Any], input_data: dict[str, Any], proof: list[str]
) -> str | None:
    try:
        expected_trace = replay(input_data, proof)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return f"invalid replay: {exc}"
    submitted_trace = result.get("trace")
    if not _trace_types_valid(submitted_trace):
        return "trace transcript mismatch"
    if submitted_trace != expected_trace:
        return "trace transcript mismatch"
    if result.get("final_expression") != input_data.get("target"):
        return "final expression mismatch"
    return None


def _mathematical_failure(result: object, input_data: dict[str, Any]) -> str | None:
    if not isinstance(result, dict):
        return "result shape mismatch"
    proof, failure = _valid_proof_and_positions(result, input_data)
    if failure is not None:
        return failure
    assert proof is not None
    return _valid_trace_and_target(result, input_data, proof)


def verify_submission(
    task_root: Path, submission: dict[str, Any], input_data: dict[str, Any]
) -> VerifyResult:
    task_id = input_data.get("task_id", "")
    envelope_ok = strict_submission_contract(
        submission,
        task_id=task_id,
        conclusion="PROOF_REPAIRED_AND_REPLAYED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}),
        verification_record="forbidden",
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") in ALLOWED_ASSURANCES
    )
    scope_accuracy = bool(
        isinstance(submission, dict)
        and submission.get("scope") == "FROZEN_METAMATH_STYLE_ASSERTION_REGISTRY"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == LIMITATIONS
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    result_shape_ok = isinstance(result, dict) and set(result) == RESULT_KEYS
    protocol_ok = bool(
        envelope_ok and scope_accuracy and assurance_ok and result_shape_ok
    )
    math_failure = _mathematical_failure(result, input_data)
    correctness = bool(math_failure is None)
    evidence_validity = bool(
        isinstance(result, dict)
        and _evidence_binds_result(submission.get("evidence"), result, task_root)
    )

    if (
        protocol_ok
        and correctness
        and evidence_validity
        and scope_accuracy
        and assurance_ok
    ):
        message = "accepted"
    elif not protocol_ok:
        message = "protocol mismatch"
    elif not correctness:
        message = math_failure or "mathematical verification failed"
    elif not evidence_validity:
        message = "evidence binding failed"
    elif not scope_accuracy:
        message = "scope or completeness mismatch"
    else:
        message = "assurance mismatch"

    return VerifyResult(
        protocol_ok,
        correctness,
        evidence_validity,
        scope_accuracy,
        assurance_ok,
        message,
    )
