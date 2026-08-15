import json
import math
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    valid_sha256_uri,
    witness_list_is_bound,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = [
    "EIGHT_STAGE_TRACE_ONLY",
    "STANDARD_PYTHAGOREAN_PARAMETERIZATION_TRUSTED",
    "NO_PROOF_ASSISTANT_VERIFICATION",
]


def expected_stage(index, m, n):
    a, b, c = 2 * m * n, m * m - n * n, m * m + n * n
    return {
        "stage": index,
        "m": m,
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "q": m * m - 2 * m * n - n * n,
        "gcd": math.gcd(m, n),
        "parity_opposite": (m - n) % 2 == 1,
    }


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(exact_value(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                exact_value(value, target)
                for value, target in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _stage_valid(stage, expected, previous_q):
    """Validate one recurrence stage against its expected values."""
    if not exact_value(stage, expected):
        return False
    if expected["gcd"] != 1 or not expected["parity_opposite"]:
        return False
    if abs(expected["q"]) != 1:
        return False
    if expected["a"] ** 2 + expected["b"] ** 2 != expected["c"] ** 2:
        return False
    if abs(expected["a"] - expected["b"]) != 1:
        return False
    return previous_q is None or expected["q"] == -previous_q


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "transform_matrix",
        "transform_determinant",
        "invariant_multiplier",
        "stages",
    }:
        return False
    if (
        not exact_value(result["transform_matrix"], [[2, 1], [1, 0]])
        or type(result["transform_determinant"]) is not int
        or result["transform_determinant"] != -1
        or type(result["invariant_multiplier"]) is not int
        or result["invariant_multiplier"] != -1
    ):
        return False
    stages = result.get("stages")
    if not isinstance(stages, list) or len(stages) != 8:
        return False
    first = stages[0]
    if not isinstance(first, dict):
        return False
    m, n = first.get("m"), first.get("n")
    if type(m) is not int or type(n) is not int:
        return False
    if not (2 <= m <= 100 and 1 <= n < m):
        return False
    previous_q = None
    for index, stage in enumerate(stages):
        expected = expected_stage(index, m, n)
        if not _stage_valid(stage, expected, previous_q):
            return False
        previous_q = expected["q"]
        m, n = 2 * m + n, m
    return True


def result_shape_valid(result):
    """Check the result has the correct keys, scalar types, and schema range
    constraints without semantic equality, so schema violations are reported
    as protocol failures rather than only as mathematical incorrectness."""
    if not isinstance(result, dict):
        return False
    if set(result) != {
        "transform_matrix",
        "transform_determinant",
        "invariant_multiplier",
        "stages",
    }:
        return False
    if (
        not isinstance(result["transform_matrix"], list)
        or len(result["transform_matrix"]) != 2
        or not all(
            isinstance(row, list)
            and len(row) == 2
            and all(type(entry) is int for entry in row)
            for row in result["transform_matrix"]
        )
    ):
        return False
    if type(result["transform_determinant"]) is not int:
        return False
    if type(result["invariant_multiplier"]) is not int:
        return False
    stages = result["stages"]
    if not isinstance(stages, list) or len(stages) != 8:
        return False
    return all(
        isinstance(s, dict)
        and set(s) == {"stage", "m", "n", "a", "b", "c", "q", "gcd", "parity_opposite"}
        and type(s["stage"]) is int
        and 0 <= s["stage"] <= 7
        and type(s["m"]) is int
        and s["m"] >= 1
        and type(s["n"]) is int
        and s["n"] >= 1
        and type(s["a"]) is int
        and s["a"] >= 1
        and type(s["b"]) is int
        and s["b"] >= 1
        and type(s["c"]) is int
        and s["c"] >= 1
        and type(s["q"]) is int
        and type(s["gcd"]) is int
        and type(s["parity_opposite"]) is bool
        for s in stages
    )


def frozen():
    return workspace_input_is_bound(W / "input.json", tests=T)


def _json_equal(left, right):
    """Compare JSON values without Python's bool/int coercion."""

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _evidence_descriptor_valid(descriptor):
    """Validate one evidence descriptor's shape, path, and digest syntax.

    File-content binding is left to the evidence metric; this checks only the
    agent-visible schema so a descriptor such as ``[null]`` or one carrying an
    extra field is reported as a protocol failure, not only as evidence
    invalidity.
    """

    return bool(
        isinstance(descriptor, dict)
        and set(descriptor) == {"path", "sha256"}
        and descriptor["path"] == "evidence/answer.txt"
        and valid_sha256_uri(descriptor["sha256"])
    )


def _evidence_descriptors_valid(evidence):
    """Require a one-element list of schema-valid evidence descriptors."""

    return bool(
        isinstance(evidence, list)
        and len(evidence) == 1
        and _evidence_descriptor_valid(evidence[0])
    )


def _consume_result_marker_char(char, prefix, pending, at_line_start):
    if char == "\n":
        marker = pending[len(prefix) :].strip() if pending.startswith(prefix) else None
        return "", True, marker
    if at_line_start and len(pending) < len(prefix):
        if char == prefix[len(pending)]:
            return pending + char, True, None
        return "", False, None
    if at_line_start:
        return pending + char, False, None
    if pending:
        return pending + char, False, None
    return pending, False, None


def _scan_result_json_markers(path):
    """Stream ``path`` and return the payload of every ``RESULT_JSON:`` line.

    The artifact is scanned incrementally in fixed-size chunks rather than
    materialized whole, so a digest-valid evidence file larger than available
    memory cannot raise ``MemoryError`` before ``reward.json`` is written.
    Non-marker lines are discarded as soon as their leading characters rule
    out the prefix, so only marker payloads are retained. Read errors fail
    closed by returning ``None``.
    """

    prefix = "RESULT_JSON:"
    max_marker_chars = 65_536
    markers: list[str] = []
    pending = ""
    at_line_start = True
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            while chunk := stream.read(65_536):
                for char in chunk:
                    pending, at_line_start, marker = _consume_result_marker_char(
                        char, prefix, pending, at_line_start
                    )
                    if marker is not None:
                        markers.append(marker)
                    if len(pending) > max_marker_chars:
                        return None
                if len(markers) > 1:
                    return markers
    except (OSError, UnicodeError):
        return None
    if pending.startswith(prefix):
        markers.append(pending[len(prefix) :].strip())
    return markers


def _evidence_bound(evidence, result):
    """Bind answer.txt to the submitted result via a RESULT_JSON line."""

    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not witness_list_is_bound(
            evidence, expected_path="evidence/answer.txt", max_bytes=None
        )
    ):
        return False
    path = resolve_evidence(
        evidence[0], expected_path="evidence/answer.txt", max_bytes=None
    )
    if path is None:
        return False
    markers = _scan_result_json_markers(path)
    if markers is None or len(markers) != 1:
        return False
    try:
        bound = json.loads(markers[0])
    except (ValueError, RecursionError):
        return False
    return json_value_equal(bound, result)


def main():
    submission = load_submission(W / "submission.json", require_input_binding=False)
    protocol_ok = submission is not None and workspace_input_is_bound(
        W / "input.json", tests=T
    )
    math_ok = bool(submission is not None and valid_result(submission.get("result")))
    evidence_ok = bool(
        protocol_ok
        and math_ok
        and _evidence_bound(submission.get("witness"), submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=evidence_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(evidence_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
