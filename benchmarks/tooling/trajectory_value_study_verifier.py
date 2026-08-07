"""Independent exact terminal verifier for the trajectory-value Codex study.

This module intentionally uses only the Python standard library.  It does not
import Jacobian, the trajectory extractor, the evaluator, a mathematical
backend, or model-runner code.  A regular submitted artifact is digest-bound
before its contents are interpreted; malformed but bound artifacts are exact
REJECTED labels, while missing or substituted inputs remain INCONCLUSIVE.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any

MAX_TASK_BYTES = 256 * 1024
MAX_SUBMISSION_BYTES = 1024 * 1024
VERIFIER_ID = "trajectory-state-value-study.clean-room-exact-verifier@1"
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_RATIONAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def object_digest(value: object) -> str:
    """Return the study's exact canonical JSON digest."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Hash one already-validated regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def verifier_digest() -> str:
    """Bind the exact independent checker implementation."""

    return file_digest(Path(__file__).resolve(strict=True))


def _regular_bounded(path: Path, maximum_bytes: int) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(status.st_mode)
        and not stat.S_ISLNK(status.st_mode)
        and status.st_size <= maximum_bytes
    )


def _load_json(path: Path, maximum_bytes: int) -> object | None:
    if not _regular_bounded(path, maximum_bytes):
        return None
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return None


def _parse_integer(value: object) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("expected canonical integer string")
    return int(value)


def _parse_rational(value: object) -> Fraction:
    if not isinstance(value, str) or _RATIONAL.fullmatch(value) is None:
        raise ValueError("expected rational string")
    return Fraction(value)


def _trim(poly: list[Fraction]) -> list[Fraction]:
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _poly(value: object) -> list[Fraction]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("polynomial coefficient array is malformed")
    return _trim([_parse_rational(item) for item in value])


def _poly_add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    size = max(len(left), len(right))
    result = [Fraction(0)] * size
    for index in range(size):
        if index < len(left):
            result[index] += left[index]
        if index < len(right):
            result[index] += right[index]
    return _trim(result)


def _poly_multiply(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return _trim(result)


def _poly_divides(dividend: list[Fraction], divisor: list[Fraction]) -> bool:
    if divisor == [Fraction(0)]:
        return False
    remainder = list(dividend)
    while remainder != [Fraction(0)] and len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        for index, coefficient in enumerate(divisor):
            remainder[index + shift] -= factor * coefficient
        _trim(remainder)
    return remainder == [Fraction(0)]


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _determinant(entries: list[list[int]]) -> int:
    dimension = len(entries)
    total = 0
    for permutation in itertools.permutations(range(dimension)):
        term = _permutation_sign(permutation)
        for row, column in enumerate(permutation):
            term *= entries[row][column]
        total += term
    return total


def _verify_integer_bezout(task: dict[str, Any], answer: object) -> dict[str, bool]:
    payload = task["payload"]
    if not isinstance(answer, dict) or set(answer) != {
        "gcd",
        "left_coefficient",
        "right_coefficient",
    }:
        return {"answer_shape": False, "exact_relation": False, "optimality": False}
    try:
        left = _parse_integer(payload["left"])
        right = _parse_integer(payload["right"])
        gcd = _parse_integer(answer["gcd"])
        left_coefficient = _parse_integer(answer["left_coefficient"])
        right_coefficient = _parse_integer(answer["right_coefficient"])
    except (KeyError, TypeError, ValueError):
        return {"answer_shape": False, "exact_relation": False, "optimality": False}
    return {
        "answer_shape": True,
        "exact_relation": (left * left_coefficient + right * right_coefficient == gcd),
        "optimality": gcd == math.gcd(left, right),
    }


def _verify_matrix_determinant(task: dict[str, Any], answer: object) -> dict[str, bool]:
    payload = task["payload"]
    if not isinstance(answer, dict) or set(answer) != {"determinant"}:
        return {"answer_shape": False, "exact_relation": False}
    try:
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list) or not 1 <= len(raw_entries) <= 7:
            raise ValueError("matrix dimension is invalid")
        entries = [
            [_parse_integer(item) for item in row]
            for row in raw_entries
            if isinstance(row, list) and len(row) == len(raw_entries)
        ]
        if len(entries) != len(raw_entries):
            raise ValueError("matrix is not square")
        determinant = _parse_integer(answer["determinant"])
    except (KeyError, TypeError, ValueError):
        return {"answer_shape": False, "exact_relation": False}
    return {
        "answer_shape": True,
        "exact_relation": determinant == _determinant(entries),
    }


def _verify_polynomial_gcd(task: dict[str, Any], answer: object) -> dict[str, bool]:
    payload = task["payload"]
    if not isinstance(answer, dict) or set(answer) != {
        "gcd",
        "left_bezout",
        "right_bezout",
    }:
        return {
            "answer_shape": False,
            "monic": False,
            "common_divisor": False,
            "bezout_identity": False,
        }
    try:
        left = _poly(payload["left_coefficients_low_to_high"])
        right = _poly(payload["right_coefficients_low_to_high"])
        gcd = _poly(answer["gcd"])
        left_bezout = _poly(answer["left_bezout"])
        right_bezout = _poly(answer["right_bezout"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {
            "answer_shape": False,
            "monic": False,
            "common_divisor": False,
            "bezout_identity": False,
        }
    return {
        "answer_shape": True,
        "monic": gcd != [Fraction(0)] and gcd[-1] == 1,
        "common_divisor": _poly_divides(left, gcd) and _poly_divides(right, gcd),
        "bezout_identity": _poly_add(
            _poly_multiply(left_bezout, left),
            _poly_multiply(right_bezout, right),
        )
        == gcd,
    }


def _verify_graph_independent_set(
    task: dict[str, Any], answer: object
) -> dict[str, bool]:
    payload = task["payload"]
    if not isinstance(answer, dict) or set(answer) != {"vertices", "optimum"}:
        return {
            "answer_shape": False,
            "witness_valid": False,
            "optimality": False,
        }
    try:
        vertices = payload["vertices"]
        edges = payload["edges"]
        selected = answer["vertices"]
        optimum = answer["optimum"]
        if (
            not isinstance(vertices, list)
            or not 1 <= len(vertices) <= 20
            or any(not isinstance(vertex, str) for vertex in vertices)
            or len(set(vertices)) != len(vertices)
            or not isinstance(edges, list)
            or not isinstance(selected, list)
            or any(not isinstance(vertex, str) for vertex in selected)
            or type(optimum) is not int
        ):
            raise ValueError("graph or answer is malformed")
        edge_set = {
            frozenset(edge)
            for edge in edges
            if isinstance(edge, list)
            and len(edge) == 2
            and all(isinstance(vertex, str) for vertex in edge)
        }
        if len(edge_set) != len(edges):
            raise ValueError("edge list is malformed")
        vertex_set = set(vertices)
        if any(len(edge) != 2 or not edge <= vertex_set for edge in edge_set):
            raise ValueError("edge endpoints are invalid")
    except (KeyError, TypeError, ValueError):
        return {
            "answer_shape": False,
            "witness_valid": False,
            "optimality": False,
        }
    selected_set = set(selected)
    witness_valid = bool(
        len(selected_set) == len(selected)
        and selected_set <= vertex_set
        and all(
            frozenset(pair) not in edge_set
            for pair in itertools.combinations(selected, 2)
        )
        and len(selected) == optimum
    )
    maximum = max(
        len(candidate)
        for size in range(len(vertices) + 1)
        for candidate in itertools.combinations(vertices, size)
        if all(
            frozenset(pair) not in edge_set
            for pair in itertools.combinations(candidate, 2)
        )
    )
    return {
        "answer_shape": True,
        "witness_valid": witness_valid,
        "optimality": optimum == maximum,
    }


_VERIFIERS = {
    "INTEGER_BEZOUT": _verify_integer_bezout,
    "MATRIX_DETERMINANT": _verify_matrix_determinant,
    "POLYNOMIAL_GCD_BEZOUT": _verify_polynomial_gcd,
    "GRAPH_MAXIMUM_INDEPENDENT_SET": _verify_graph_independent_set,
}


def verify_workspace(expected_task: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Verify one isolated workspace and bind its exact input and submission."""

    task_path = workspace / "task.json"
    submission_path = workspace / "submission.json"
    actual_task = _load_json(task_path, MAX_TASK_BYTES)
    input_binding_valid = bool(
        actual_task == expected_task
        and object_digest(actual_task) == object_digest(expected_task)
    )
    submission_regular = _regular_bounded(submission_path, MAX_SUBMISSION_BYTES)
    submission_digest = file_digest(submission_path) if submission_regular else None
    submission = (
        _load_json(submission_path, MAX_SUBMISSION_BYTES)
        if submission_regular
        else None
    )
    artifact_binding_valid = submission_regular
    checks: dict[str, bool] = {
        "input_binding": input_binding_valid,
        "artifact_binding": artifact_binding_valid,
        "submission_json": isinstance(submission, dict),
        "task_identity": bool(
            isinstance(submission, dict)
            and submission.get("task_id") == expected_task.get("task_id")
        ),
        "envelope_shape": bool(
            isinstance(submission, dict) and set(submission) == {"task_id", "answer"}
        ),
    }
    if input_binding_valid and isinstance(submission, dict):
        verifier = _VERIFIERS.get(str(expected_task.get("kind")))
        if verifier is not None:
            checks.update(verifier(expected_task, submission.get("answer")))
    mathematical_checks = tuple(
        value
        for name, value in checks.items()
        if name not in {"input_binding", "artifact_binding"}
    )
    if not input_binding_valid or not artifact_binding_valid:
        acceptance = "INCONCLUSIVE"
    else:
        acceptance = (
            "ACCEPTED"
            if mathematical_checks and all(mathematical_checks)
            else "REJECTED"
        )
    return {
        "schema_version": "1",
        "verifier_id": VERIFIER_ID,
        "verifier_digest": verifier_digest(),
        "clean_room": True,
        "verifier_execution_status": "COMPLETED",
        "acceptance": acceptance,
        "input_binding_valid": input_binding_valid,
        "artifact_binding_valid": artifact_binding_valid,
        "task_digest": object_digest(expected_task),
        "submission_digest": submission_digest,
        "checks": checks,
    }


__all__ = [
    "MAX_SUBMISSION_BYTES",
    "MAX_TASK_BYTES",
    "VERIFIER_ID",
    "file_digest",
    "object_digest",
    "verifier_digest",
    "verify_workspace",
]
