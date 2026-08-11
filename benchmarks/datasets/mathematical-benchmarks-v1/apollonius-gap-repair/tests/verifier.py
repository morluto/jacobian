import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    _public_submission_is_valid,
    false_verified_claim,
    is_regular_bounded_file,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
LIMITATION = "The certificate repairs the annotated coordinate identity; it does not independently formalize every theorem or endpoint convention in the full geometry proof."
RATIONAL = re.compile(r"^-?(?:0|[1-9][0-9]{0,5})(?:/[1-9][0-9]{0,5})?$")


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _submission() -> dict[str, Any] | None:
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _q(value: object) -> Fraction | None:
    if not isinstance(value, str) or RATIONAL.fullmatch(value) is None:
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


def _qs(value: object) -> list[Fraction] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    parsed = [_q(item) for item in value]
    return None if any(item is None for item in parsed) else parsed


def _result(value: object, frozen: dict[str, Any]) -> bool:
    fields = {
        "k",
        "c",
        "p",
        "q",
        "center",
        "radius",
        "circle_coefficients",
        "distance_coefficients",
        "multiplier",
    }
    if (
        not isinstance(value, dict)
        or not fields.issubset(value)
        or frozen.get("human_score") != 0
    ):
        return False
    k, c, p, q, center, radius, multiplier = (
        _q(value[name])
        for name in ("k", "c", "p", "q", "center", "radius", "multiplier")
    )
    if None in {k, c, p, q, center, radius, multiplier} or not (
        k > 0 and c > 0 and k != 1
    ):
        return False
    if (
        p != k * c / (k + 1)
        or q != k * c / (k - 1)
        or center != (p + q) / 2
        or radius != abs(q - p) / 2
        or radius <= 0
    ):
        return False
    circle, distance = (
        _qs(value["circle_coefficients"]),
        _qs(value["distance_coefficients"]),
    )
    expected_circle = [
        Fraction(1),
        Fraction(1),
        -2 * center,
        center * center - radius * radius,
    ]
    expected_distance = [1 - k * k, 1 - k * k, 2 * k * k * c, -k * k * c * c]
    return (
        circle == expected_circle
        and distance == expected_distance
        and multiplier == 1 - k * k
        and distance == [multiplier * x for x in circle]
    )


def _result_values_protocol_valid(value: object) -> bool:
    fields = {
        "k",
        "c",
        "p",
        "q",
        "center",
        "radius",
        "circle_coefficients",
        "distance_coefficients",
        "multiplier",
    }
    if not isinstance(value, dict) or not fields.issubset(value):
        return False
    scalar_names = fields - {"circle_coefficients", "distance_coefficients"}
    scalars = {name: _q(value[name]) for name in scalar_names}
    if any(parsed is None for parsed in scalars.values()):
        return False
    if not (
        scalars["k"] > 0
        and scalars["k"] != 1
        and scalars["c"] > 0
        and scalars["radius"] > 0
    ):
        return False
    return all(
        isinstance(value[name], list)
        and len(value[name]) == 4
        and all(_q(item) is not None for item in value[name])
        for name in ("circle_coefficients", "distance_coefficients")
    )


def _result_protocol_valid(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "k",
            "c",
            "p",
            "q",
            "center",
            "radius",
            "circle_coefficients",
            "distance_coefficients",
            "multiplier",
        }
        and _result_values_protocol_valid(value)
    )


def _encode_certificate_lines(expected: list[str]) -> tuple[bytes, ...] | None:
    try:
        return tuple(line.encode() for line in expected)
    except UnicodeError:
        return None


def _stream_matches_certificate(path: Path, expected: list[str]) -> bool:
    """Compare exact certificate lines without materializing the artifact."""

    expected_bytes = _encode_certificate_lines(expected)
    if expected_bytes is None:
        return False
    max_line_bytes = max(map(len, expected_bytes), default=0) + 2
    line_index = 0
    try:
        with path.open("rb") as stream:
            while raw := stream.readline(max_line_bytes):
                if len(raw) == max_line_bytes and not raw.endswith(b"\n"):
                    return False
                line = raw[:-1] if raw.endswith(b"\n") else raw
                if line.endswith(b"\r"):
                    line = line[:-1]
                if (
                    line_index >= len(expected_bytes)
                    or line != expected_bytes[line_index]
                ):
                    return False
                line_index += 1
    except (OSError, UnicodeError):
        return False
    return line_index == len(expected_bytes)


def _evidence(value: object, result: object) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    if not isinstance(result, dict):
        return False
    if not _result_values_protocol_valid(result):
        return False
    circle = result.get("circle_coefficients")
    distance = result.get("distance_coefficients")
    multiplier = result.get("multiplier")
    if not all(
        isinstance(items, list)
        and len(items) == 4
        and all(isinstance(item, str) for item in items)
        for items in (circle, distance)
    ) or not isinstance(multiplier, str):
        return False
    certificate = [
        "apollonius-coefficient-certificate-v1",
        f"multiplier: {multiplier}",
        "circle_coefficients: " + ",".join(circle),
        "distance_coefficients: " + ",".join(distance),
    ]
    encoded_certificate = _encode_certificate_lines(certificate)
    if encoded_certificate is None:
        return False
    max_bytes = sum(len(line) + 1 for line in encoded_certificate)
    path = resolve_evidence(
        value[0], expected_path="evidence/answer.txt", max_bytes=max_bytes
    )
    if path is None:
        return False
    return _stream_matches_certificate(path, certificate)


def main() -> None:
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    expected = json.loads((TESTS / "expected.json").read_text())
    envelope_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        min_limitations=1,
        verification_record="forbidden",
    )
    contract = bool(
        envelope_contract
        and _public_submission_is_valid(submission)
        and _result_protocol_valid(data.get("result"))
    )
    math_correct = _result(data.get("result"), _load())
    evidence_valid = _evidence(data.get("evidence"), data.get("result"))
    scope_correct = data.get("scope") == expected["required_scope"]
    assurance_correct = data.get("claimed_assurance") == expected["maximum_assurance"]
    limitations = data.get("limitations")
    limitations_correct = limitations == [LIMITATION]
    completeness_correct = data.get("completeness") == "COMPLETE"
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        input_bound
        and contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and completeness_correct
        and limitations_correct
        and not false_certification
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "input_binding": float(input_bound),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "completeness_accuracy": float(completeness_correct),
                "limitations_accuracy": float(limitations_correct),
                "protocol_compliance": float(contract),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
