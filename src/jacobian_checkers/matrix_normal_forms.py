"""Independent exact replay for integer row Hermite normal forms.

This checker intentionally uses only the Python standard library. It does not
import Jacobian contracts, Python-FLINT, SymPy, or producer code.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from jacobian_checkers.bound_artifacts import valid_unscoped_unencoded_bindings

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_MATRIX_BINDING_KEYS = {
    "binding_version",
    "matrix_artifact_uri",
    "matrix_object_digest",
    "matrix_payload_digest",
    "row_count",
    "column_count",
}
_PROVIDER_KEYS = {
    "runtime_version",
    "provider",
    "availability",
    "version",
    "digest",
    "digest_kind",
    "platform",
    "install_tier",
    "license_id",
    "license_files",
    "features",
    "checker_ids",
    "configuration",
    "distribution_import_name",
    "distribution_required_attributes",
    "diagnostic",
}
MAX_MATRIX_DIMENSION = 32
MAX_INTEGER_DIGITS = 256
_HNF_CONFIGURATION = {
    "distribution": "python-flint",
    "domain": "ZZ",
    "operation": "fmpz_mat.hnf(transform=True)",
    "flint_library_version": "3.6.0",
    "maximum_rows": 32,
    "maximum_columns": 32,
    "normal_form_convention": "FLINT_ROW_HNF",
    "relation": "H=U*A",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_artifact(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        return False
    if (
        not isinstance(value["artifact_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["artifact_uri"]) is None
        or not isinstance(value["object_digest"], str)
        or _DIGEST.fullmatch(value["object_digest"]) is None
        or not isinstance(value["payload_digest"], str)
        or _DIGEST.fullmatch(value["payload_digest"]) is None
        or not isinstance(value["schema_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["schema_uri"]) is None
        or not isinstance(value["semantics_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["semantics_uri"]) is None
    ):
        return False
    parents = value["parents"]
    return (
        isinstance(parents, list)
        and len(parents) == len(set(parents))
        and all(
            isinstance(parent, str) and _ARTIFACT_URI.fullmatch(parent) is not None
            for parent in parents
        )
    )


def _integer(value: object) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > MAX_INTEGER_DIGITS
    ):
        raise ValueError("matrix entry is not a bounded canonical integer")
    result = int(value)
    if str(result) != value:
        raise ValueError("matrix entry is not canonical")
    return result


def _matrix(payload: object) -> list[list[int]]:
    if not isinstance(payload, dict) or set(payload) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("integer matrix has an invalid shape")
    if payload["matrix_schema_version"] != "1" or payload["domain"] != "ZZ":
        raise ValueError("integer matrix uses unsupported semantics")
    entries = payload["entries"]
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_MATRIX_DIMENSION
        or not isinstance(entries[0], list)
        or not 1 <= len(entries[0]) <= MAX_MATRIX_DIMENSION
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise ValueError("integer matrix dimensions are malformed")
    return [[_integer(value) for value in row] for row in entries]


def _validate_candidate(
    payload: object,
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    rows: int,
    columns: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if not isinstance(payload, dict) or set(payload) != {
        "normal_form_schema_version",
        "source",
        "declared_scope",
        "normal_form",
        "transformation",
        "producer",
        "resource_budget",
        "method",
    }:
        raise ValueError("HNF candidate has an invalid shape")
    if (
        payload["normal_form_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_MATRIX"
        or payload["method"] != "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM"
    ):
        raise ValueError("HNF candidate uses unsupported semantics")
    binding = payload["source"]
    if not isinstance(binding, dict) or set(binding) != _MATRIX_BINDING_KEYS:
        raise ValueError("HNF source binding is malformed")
    if binding != {
        "binding_version": "1",
        "matrix_artifact_uri": claim["artifact_uri"],
        "matrix_object_digest": claim["object_digest"],
        "matrix_payload_digest": claim["payload_digest"],
        "row_count": rows,
        "column_count": columns,
    }:
        raise ValueError("HNF candidate is not exactly bound to the source matrix")
    if claim["artifact_uri"] not in candidate["parents"]:
        raise ValueError("HNF candidate is missing source-matrix lineage")
    normal_form = _matrix(payload["normal_form"])
    transformation = _matrix(payload["transformation"])
    if (
        len(normal_form) != rows
        or len(normal_form[0]) != columns
        or len(transformation) != rows
        or len(transformation[0]) != rows
    ):
        raise ValueError("HNF candidate dimensions do not match the source")
    provider = payload["producer"]
    if (
        not isinstance(provider, dict)
        or set(provider) != _PROVIDER_KEYS
        or provider["runtime_version"] != "1"
        or provider["provider"] != "python-flint"
        or provider["availability"] != "AVAILABLE"
        or provider["version"] != "0.9.0"
        or not isinstance(provider["digest"], str)
        or _DIGEST.fullmatch(provider["digest"]) is None
        or provider["digest_kind"] != "PYTHON_DISTRIBUTION_RECORD"
        or provider["install_tier"] != "T1"
        or provider["configuration"] != _HNF_CONFIGURATION
    ):
        raise ValueError("HNF producer identity is malformed")
    budget = payload["resource_budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"budget_version", "wall_seconds"}
        or budget["budget_version"] != "1"
        or type(budget["wall_seconds"]) is not int
        or not 1 <= budget["wall_seconds"] <= 60
    ):
        raise ValueError("HNF resource budget is malformed")
    return normal_form, transformation


def _multiply(
    left: list[list[int]],
    right: list[list[int]],
) -> list[list[int]]:
    return [
        [
            sum(
                left_value * right[row][column]
                for row, left_value in enumerate(left_row)
            )
            for column in range(len(right[0]))
        ]
        for left_row in left
    ]


def _determinant_bareiss(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 1:
        return matrix[0][0]
    work = [row[:] for row in matrix]
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign = -sign
        pivot = work[column][column]
        for row in range(column + 1, size):
            for target_column in range(column + 1, size):
                numerator = (
                    work[row][target_column] * pivot
                    - work[row][column] * work[column][target_column]
                )
                quotient, remainder = divmod(numerator, previous_pivot)
                if remainder != 0:
                    raise ValueError("fraction-free determinant division failed")
                work[row][target_column] = quotient
            work[row][column] = 0
        previous_pivot = pivot
    return sign * work[-1][-1]


def _is_row_hnf(matrix: list[list[int]]) -> bool:
    last_nonzero_row = -1
    for row, values in enumerate(matrix):
        if any(value != 0 for value in values):
            last_nonzero_row = row
    previous_pivot_column = -1
    for row in range(last_nonzero_row + 1):
        pivot_column = next(
            (column for column, value in enumerate(matrix[row]) if value != 0),
            None,
        )
        if pivot_column is None or pivot_column <= previous_pivot_column:
            return False
        pivot = matrix[row][pivot_column]
        if pivot < 0:
            return False
        if any(
            matrix[above][pivot_column] < 0 or matrix[above][pivot_column] >= pivot
            for above in range(row)
        ):
            return False
        previous_pivot_column = pivot_column
    return all(
        all(value == 0 for value in matrix[row])
        for row in range(last_nonzero_row + 1, len(matrix))
    )


def check_hermite_normal_form(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only exact row-HNF evidence with a unimodular left transform."""

    try:
        if not isinstance(request, dict) or set(request) != {
            "request_version",
            "claim",
            "candidate",
            "scope",
            "witness",
            "expected_bindings",
        }:
            return _reject("malformed checker request")
        if request["request_version"] != "1" or request["scope"] is not None:
            return _reject("unsupported checker request")
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        if not all(_valid_artifact(item) for item in (claim, candidate, witness)):
            return _reject("checker artifact metadata is malformed")
        expected_bindings = request["expected_bindings"]
        if not valid_unscoped_unencoded_bindings(expected_bindings):
            return _reject("expected evidence bindings are malformed")
        if (
            claim["semantics_uri"] != candidate["semantics_uri"]
            or claim["semantics_uri"] != witness["semantics_uri"]
        ):
            return _reject("checker artifacts use different semantics")
        for artifact, label in (
            (claim, "source matrix"),
            (candidate, "HNF candidate"),
            (witness, "HNF witness"),
        ):
            if artifact["payload_digest"] != _sha256(
                _canonical_json(artifact["payload"])
            ):
                return _reject(f"{label} payload digest does not match")

        source = _matrix(claim["payload"])
        normal_form, transformation = _validate_candidate(
            candidate["payload"],
            claim=claim,
            candidate=candidate,
            rows=len(source),
            columns=len(source[0]),
        )
        if (
            expected_bindings["claim_digest"] != claim["object_digest"]
            or expected_bindings["candidate_digest"] != candidate["object_digest"]
        ):
            return _reject("expected evidence bindings do not match artifacts")
        envelope = witness["payload"]
        if not isinstance(envelope, dict) or set(envelope) != {
            "evidence_schema_version",
            "witness_format",
            "format_version",
            "role",
            "bindings",
            "payload",
        }:
            return _reject("HNF witness envelope is malformed")
        if (
            envelope["evidence_schema_version"] != "1"
            or envelope["witness_format"] != "matrix.normal_form.hermite"
            or envelope["format_version"] != "1"
            or envelope["role"] != "SUPPORTS_CLAIM"
            or envelope["bindings"] != expected_bindings
        ):
            return _reject("HNF witness envelope is not exactly bound")
        if envelope["payload"] != {
            "matrix_uri": claim["artifact_uri"],
            "normal_form_uri": candidate["artifact_uri"],
        }:
            return _reject("HNF witness points at different artifacts")
        if not {
            claim["artifact_uri"],
            candidate["artifact_uri"],
        }.issubset(set(witness["parents"])):
            return _reject("HNF witness is missing required lineage")

        if _multiply(transformation, source) != normal_form:
            return _reject("the proposed exact relation H = U A does not hold")
        if abs(_determinant_bareiss(transformation)) != 1:
            return _reject("the proposed left transformation is not unimodular")
        if not _is_row_hnf(normal_form):
            return _reject("the candidate does not satisfy FLINT row-HNF conditions")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"checked H = U A, det(U) = +/-1, and row-HNF conditions for "
                f"the full {len(source)} by {len(source[0])} integer matrix"
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed checker request")
