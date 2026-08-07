"""Independent exact determinant replay over rational matrices.

This checker intentionally uses only the Python standard library. It does not
import Jacobian contracts, SymPy, or determinant producer code.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
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
_MAX_DIMENSION = 32


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
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
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("rational component is not a canonical integer")
    result = int(value)
    if str(result) != value:
        raise ValueError("rational component is not canonical")
    return result


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational value has an invalid shape")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError("rational value is not reduced")
    return result


def _matrix(payload: object) -> list[list[Fraction]]:
    if not isinstance(payload, dict) or set(payload) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("rational matrix has an invalid shape")
    if payload["matrix_schema_version"] != "1" or payload["domain"] != "QQ":
        raise ValueError("rational matrix uses unsupported semantics")
    entries = payload["entries"]
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= _MAX_DIMENSION
        or not isinstance(entries[0], list)
        or len(entries[0]) != len(entries)
        or any(not isinstance(row, list) or len(row) != len(entries) for row in entries)
    ):
        raise ValueError("determinant requires a bounded square matrix")
    return [[_rational(value) for value in row] for row in entries]


def _candidate(
    payload: object,
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> Fraction:
    if not isinstance(payload, dict) or set(payload) != {
        "result_schema_version",
        "matrix_uri",
        "determinant",
        "method",
        "backend",
        "backend_version",
    }:
        raise ValueError("determinant candidate has an invalid shape")
    if (
        payload["result_schema_version"] != "1"
        or payload["matrix_uri"] != claim["artifact_uri"]
        or payload["method"] != "FRACTION_FREE_BAREISS"
        or payload["backend"] != "sympy"
        or not isinstance(payload["backend_version"], str)
        or not payload["backend_version"]
        or candidate["parents"] != [claim["artifact_uri"]]
    ):
        raise ValueError("determinant candidate is not bound to the source matrix")
    return _rational(payload["determinant"])


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    """Exact Gaussian-elimination determinant with explicit row swaps."""

    work = [row[:] for row in matrix]
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot_row = next(
            (row for row in range(column, len(work)) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            determinant = -determinant
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, len(work)):
            multiplier = work[row][column] / pivot
            for target_column in range(column + 1, len(work)):
                work[row][target_column] -= multiplier * work[column][target_column]
            work[row][column] = Fraction(0)
    return determinant


_DET_REQUEST_KEYS = {
    "request_version",
    "claim",
    "candidate",
    "scope",
    "witness",
    "expected_bindings",
}
_DET_WITNESS_ENVELOPE_KEYS = {
    "evidence_schema_version",
    "witness_format",
    "format_version",
    "role",
    "bindings",
    "payload",
}


def _check_det_request_envelope(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != _DET_REQUEST_KEYS:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _check_det_artifacts(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if not all(_valid_artifact(item) for item in (claim, candidate, witness)):
        return "checker artifact metadata is malformed"
    if not valid_unscoped_unencoded_bindings(expected_bindings):
        return "expected evidence bindings are malformed"
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != witness["semantics_uri"]
    ):
        return "checker artifacts use different semantics"
    return None


def _check_det_digests(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
) -> str | None:
    for artifact, label in (
        (claim, "source matrix"),
        (candidate, "determinant candidate"),
        (witness, "determinant witness"),
    ):
        if artifact["payload_digest"] != _sha256(_canonical_json(artifact["payload"])):
            return f"{label} payload digest does not match"
    return None


def _check_det_binding_match(
    expected_bindings: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if (
        expected_bindings["claim_digest"] != claim["object_digest"]
        or expected_bindings["candidate_digest"] != candidate["object_digest"]
    ):
        return "expected evidence bindings do not match artifacts"
    return None


def _check_det_witness_envelope(
    envelope: object,
    expected_bindings: object,
) -> str | None:
    if not isinstance(envelope, dict) or set(envelope) != _DET_WITNESS_ENVELOPE_KEYS:
        return "determinant witness envelope is malformed"
    if (
        envelope["evidence_schema_version"] != "1"
        or envelope["witness_format"] != "matrix.rational_determinant"
        or envelope["format_version"] != "1"
        or envelope["role"] != "SUPPORTS_CLAIM"
        or envelope["bindings"] != expected_bindings
    ):
        return "determinant witness envelope is not exactly bound"
    return None


def _check_det_witness_payload(
    witness: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if witness["payload"]["payload"] != {
        "matrix_uri": claim["artifact_uri"],
        "determinant_uri": candidate["artifact_uri"],
    }:
        return "determinant witness points at different artifacts"
    if len(witness["parents"]) != 2 or set(witness["parents"]) != {
        claim["artifact_uri"],
        candidate["artifact_uri"],
    }:
        return "determinant witness is missing required lineage"
    return None


def _check_det_witness(
    witness: dict[str, Any],
    expected_bindings: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    error = _check_det_witness_envelope(witness["payload"], expected_bindings)
    if error is not None:
        return error
    return _check_det_witness_payload(witness, claim, candidate)


def check_rational_determinant(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a fully bound candidate equal to an exact independent replay."""

    try:
        error = _check_det_request_envelope(request)
        if error is not None:
            return _reject(error)
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        expected_bindings = request["expected_bindings"]
        error = _check_det_artifacts(claim, candidate, witness, expected_bindings)
        if error is not None:
            return _reject(error)
        error = _check_det_digests(claim, candidate, witness)
        if error is not None:
            return _reject(error)

        matrix = _matrix(claim["payload"])
        declared = _candidate(
            candidate["payload"],
            claim=claim,
            candidate=candidate,
        )
        error = _check_det_binding_match(expected_bindings, claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_det_witness(witness, expected_bindings, claim, candidate)
        if error is not None:
            return _reject(error)

        computed = _determinant(matrix)
        if computed != declared:
            return _reject("declared determinant does not match exact recomputation")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"recomputed the determinant exactly for the full "
                f"{len(matrix)} by {len(matrix)} rational matrix"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed checker request")


def _check_rank_request_envelope(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != _DET_REQUEST_KEYS:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _check_rank_artifacts_and_bindings(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
    bindings: object,
) -> str | None:
    if not all(_valid_artifact(item) for item in (claim, candidate, witness)):
        return "checker artifact metadata is malformed"
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != witness["semantics_uri"]
        or any(
            artifact["payload_digest"] != _sha256(_canonical_json(artifact["payload"]))
            for artifact in (claim, candidate, witness)
        )
    ):
        return "rank artifacts have changed payloads or semantics"
    if (
        not valid_unscoped_unencoded_bindings(bindings)
        or not isinstance(bindings, dict)
        or bindings["claim_digest"] != claim["object_digest"]
        or bindings["candidate_digest"] != candidate["object_digest"]
        or len(witness["parents"]) != 2
        or set(witness["parents"]) != {claim["artifact_uri"], candidate["artifact_uri"]}
    ):
        return "rank evidence bindings or lineage do not match"
    return None


def _check_rank_matrix(
    claim: dict[str, Any],
) -> list[list[Fraction]] | str:
    payload = claim["payload"]
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"matrix_schema_version", "domain", "entries"}
        or payload["matrix_schema_version"] != "1"
        or payload["domain"] != "QQ"
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= _MAX_DIMENSION
        or not isinstance(entries[0], list)
        or not 1 <= len(entries[0]) <= _MAX_DIMENSION
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        return "rank source matrix is malformed"
    return [[_rational(value) for value in row] for row in entries]


def _check_rank_candidate(
    declared: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if (
        not isinstance(declared, dict)
        or set(declared)
        != {
            "result_schema_version",
            "matrix_uri",
            "rank",
            "pivot_columns",
            "method",
            "backend",
            "backend_version",
        }
        or declared["result_schema_version"] != "1"
        or declared["matrix_uri"] != claim["artifact_uri"]
        or candidate["parents"] != [claim["artifact_uri"]]
        or declared["method"] != "EXACT_RATIONAL_ROW_REDUCTION"
        or declared["backend"] != "sympy"
        or not isinstance(declared["rank"], int)
        or isinstance(declared["rank"], bool)
    ):
        return "rank candidate is malformed or misbound"
    return None


def _check_rank_witness(
    envelope: object,
    bindings: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if (
        not isinstance(envelope, dict)
        or envelope.get("witness_format") != "matrix.rational_rank"
        or envelope.get("format_version") != "1"
        or envelope.get("role") != "SUPPORTS_CLAIM"
        or envelope.get("bindings") != bindings
        or envelope.get("payload")
        != {
            "matrix_uri": claim["artifact_uri"],
            "rank_uri": candidate["artifact_uri"],
        }
    ):
        return "rank witness is malformed or misbound"
    return None


def _compute_rank(matrix: list[list[Fraction]]) -> int:
    rank = 0
    column_count = len(matrix[0])
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        for row in range(rank + 1, len(matrix)):
            multiplier = matrix[row][column] / pivot_value
            for target in range(column, column_count):
                matrix[row][target] -= multiplier * matrix[rank][target]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def check_rational_rank(request: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute the rank of one bound rectangular QQ matrix."""

    try:
        error = _check_rank_request_envelope(request)
        if error is not None:
            return _reject(error)
        claim, candidate, witness = (
            request["claim"],
            request["candidate"],
            request["witness"],
        )
        bindings = request["expected_bindings"]
        error = _check_rank_artifacts_and_bindings(claim, candidate, witness, bindings)
        if error is not None:
            return _reject(error)
        result = _check_rank_matrix(claim)
        if isinstance(result, str):
            return _reject(result)
        matrix = result
        declared = candidate["payload"]
        error = _check_rank_candidate(declared, claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_rank_witness(witness["payload"], bindings, claim, candidate)
        if error is not None:
            return _reject(error)
        rank = _compute_rank(matrix)
        if rank != declared["rank"]:
            return _reject("declared rank does not match exact recomputation")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": f"recomputed exact rank {rank} for the full rectangular matrix",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed checker request")
