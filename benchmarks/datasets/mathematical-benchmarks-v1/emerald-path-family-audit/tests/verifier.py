import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    resolve_evidence,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

TESTS = Path("/tests")
LIMITATION = "The certificate refutes the published singleton claim and proves sufficiency for its submitted family member; it does not independently prove necessity for every possible trip."
RATIONAL = re.compile(r"^-?(?:0|[1-9][0-9]{0,63})(?:/[1-9][0-9]{0,63})?$")
RESULT_FIELDS = frozenset({"alpha", "beta", "even_offset", "odd_offset", "trace"})
TRACE_FIELDS = frozenset({"n", "x", "y", "value", "floor"})


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if type(value) is not str or RATIONAL.fullmatch(value) is None:
        return None
    try:
        parsed = Fraction(value)
    except (MemoryError, OverflowError, ValueError, ZeroDivisionError):
        return None
    return parsed


def _result(value: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or not RESULT_FIELDS.issubset(value):
        return False
    alpha, beta = _fraction(value["alpha"]), _fraction(value["beta"])
    even, odd = _fraction(value["even_offset"]), _fraction(value["odd_offset"])
    if None in {alpha, beta, even, odd} or not (alpha > beta > 0 and alpha + beta == 2):
        return False
    if even != 0 or odd != (alpha - beta) / 2 or not (0 <= odd < 1):
        return False
    trace = value["trace"]
    length, band = frozen.get("trace_length"), frozen.get("band")
    if (
        type(length) is not int
        or type(band) is not int
        or not isinstance(trace, list)
        or len(trace) != length
    ):
        return False
    if any(
        not isinstance(item, dict)
        or not TRACE_FIELDS.issubset(item)
        or any(type(item[field]) is not int for field in ("n", "x", "y", "floor"))
        or not isinstance(item["value"], str)
        for item in trace
    ):
        return False
    for n in range(length):
        x, y = (n + 1) // 2, n // 2
        exact = x * alpha + y * beta
        item = trace[n]
        if (
            item["n"] != n
            or item["x"] != x
            or item["y"] != y
            or _fraction(item["value"]) != exact
            or item["floor"] != exact.numerator // exact.denominator
        ):
            return False
        if abs(x - y) >= band:
            return False
    return all(item["floor"] == item["n"] for item in trace)


def _result_protocol_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        return False
    if not all(
        isinstance(value[name], str) and _fraction(value[name]) is not None
        for name in ("alpha", "beta", "even_offset", "odd_offset")
    ):
        return False
    trace = value["trace"]
    return bool(
        isinstance(trace, list)
        and len(trace) == 16
        and all(
            isinstance(item, dict)
            and set(item) == TRACE_FIELDS
            and all(type(item[field]) is int for field in ("n", "x", "y", "floor"))
            and isinstance(item["value"], str)
            and _fraction(item["value"]) is not None
            for item in trace
        )
    )


def _stream_matches_certificate(path: Path, expected: list[str]) -> bool:
    """Compare nonempty stripped lines without materializing the artifact."""

    expected_bytes = tuple(line.encode() for line in expected)
    max_line_bytes = max(map(len, expected_bytes), default=0) + 2
    line_index = 0
    try:
        with path.open("rb") as stream:
            while raw := stream.readline(max_line_bytes):
                if len(raw) == max_line_bytes and not raw.endswith(b"\n"):
                    return False
                line = raw.strip()
                if not line:
                    continue
                if (
                    line_index >= len(expected_bytes)
                    or line != expected_bytes[line_index]
                ):
                    return False
                line_index += 1
    except (OSError, UnicodeError):
        return False
    return line_index == len(expected_bytes)


def _witness(value: object, result: object) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    if not isinstance(result, dict):
        return False
    trace = result.get("trace")
    if (
        not isinstance(trace, list)
        or len(trace) != 16
        or any(
            not isinstance(item, dict)
            or set(item) != TRACE_FIELDS
            or any(type(item[field]) is not int for field in ("n", "x", "y", "floor"))
            or not isinstance(item["value"], str)
            or _fraction(item["value"]) is None
            for item in trace
        )
    ):
        return False
    if not all(
        isinstance(result.get(name), str) and _fraction(result[name]) is not None
        for name in ("alpha", "beta", "even_offset", "odd_offset")
    ) or not isinstance(result.get("trace"), list):
        return False
    try:
        trace_digest = hashlib.sha256(
            json.dumps(result["trace"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    except (TypeError, ValueError, RecursionError, MemoryError):
        return False
    certificate = [
        "emerald-path-family-certificate-v1",
        f"alpha: {result.get('alpha')}",
        f"beta: {result.get('beta')}",
        f"even_offset: {result.get('even_offset')}",
        f"odd_offset: {result.get('odd_offset')}",
        f"trace_sha256: {trace_digest}",
    ]
    max_bytes = sum(len(line.encode()) + 1 for line in certificate)
    path = resolve_evidence(
        value[0], expected_path="evidence/answer.txt", max_bytes=max_bytes
    )
    return path is not None and _stream_matches_certificate(path, certificate)


def main() -> None:
    submission = load_submission_raw(require_input_binding=False)
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    contract = bool(
        submission_matches_public_schema(submission)
        and _result_protocol_valid(data.get("result"))
    )
    math_correct = _result(data.get("result"), _load())
    witness_valid = _witness(data.get("witness"), data.get("result"))
    correct = input_bound and contract and math_correct and witness_valid
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(witness_valid),
                "input_binding": float(input_bound),
                "protocol_compliance": float(contract),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
