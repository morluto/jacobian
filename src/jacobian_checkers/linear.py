"""Independent exact replay for rational linear-solution witnesses.

This module intentionally uses only the Python standard library. It does not
import Jacobian contracts, Python-FLINT, SymPy, or any producer implementation.
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
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_SYSTEM_BINDING_KEYS = {
    "binding_version",
    "system_artifact_uri",
    "system_object_digest",
    "system_payload_digest",
    "variable_order_digest",
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
MAX_LINEAR_DIMENSION = 32
MAX_RATIONAL_DIGITS = 256


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
    # Linear artifacts contain only ASCII strings, lists, and maps. For that
    # subset this is byte-identical to Jacobian's canonical JSON encoder.
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


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational value has an invalid shape")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
        or len(numerator.lstrip("-")) > MAX_RATIONAL_DIGITS
        or len(denominator.lstrip("-")) > MAX_RATIONAL_DIGITS
    ):
        raise ValueError("rational value is not bounded canonical integer data")
    result = Fraction(int(numerator), int(denominator))
    if str(result.numerator) != numerator or str(result.denominator) != denominator:
        raise ValueError("rational value is not reduced and canonical")
    return result


def _validate_system(
    payload: object,
) -> tuple[list[list[Fraction]], list[Fraction], list[str]]:
    if not isinstance(payload, dict) or set(payload) != {
        "system_schema_version",
        "domain",
        "relation",
        "variables",
        "coefficients",
        "rhs",
    }:
        raise ValueError("rational linear system has an invalid shape")
    if (
        payload["system_schema_version"] != "1"
        or payload["domain"] != "QQ"
        or payload["relation"] != "AX_EQUALS_B"
    ):
        raise ValueError("rational linear system uses unsupported semantics")
    variables = payload["variables"]
    if (
        not isinstance(variables, list)
        or not 1 <= len(variables) <= MAX_LINEAR_DIMENSION
        or any(
            not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
            for variable in variables
        )
        or len(variables) != len(set(variables))
    ):
        raise ValueError("rational linear-system variables are malformed")
    matrix = payload["coefficients"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("rational coefficient matrix has an invalid shape")
    if matrix["matrix_schema_version"] != "1" or matrix["domain"] != "QQ":
        raise ValueError("rational coefficient matrix uses unsupported semantics")
    entries = matrix["entries"]
    rhs = payload["rhs"]
    if (
        not isinstance(entries, list)
        or not isinstance(rhs, list)
        or not 1 <= len(entries) <= MAX_LINEAR_DIMENSION
        or len(entries) != len(rhs)
        or any(
            not isinstance(row, list) or len(row) != len(variables) for row in entries
        )
    ):
        raise ValueError("rational linear-system dimensions do not match")
    return (
        [[_rational(value) for value in row] for row in entries],
        [_rational(value) for value in rhs],
        variables,
    )


def _validate_solution(
    payload: object,
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    rows: int,
    columns: int,
    variables: list[str],
) -> list[Fraction]:
    if not isinstance(payload, dict) or set(payload) != {
        "solution_schema_version",
        "system",
        "declared_scope",
        "values",
        "producer",
        "resource_budget",
        "method",
    }:
        raise ValueError("rational solution has an invalid shape")
    if (
        payload["solution_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_SYSTEM"
        or payload["method"] != "RREF_FREE_VARIABLES_ZERO"
    ):
        raise ValueError("rational solution uses unsupported semantics")
    binding = payload["system"]
    if not isinstance(binding, dict) or set(binding) != _SYSTEM_BINDING_KEYS:
        raise ValueError("rational solution system binding is malformed")
    expected_variable_digest = _sha256(_canonical_json({"variables": variables}))
    if binding != {
        "binding_version": "1",
        "system_artifact_uri": claim["artifact_uri"],
        "system_object_digest": claim["object_digest"],
        "system_payload_digest": claim["payload_digest"],
        "variable_order_digest": expected_variable_digest,
        "row_count": rows,
        "column_count": columns,
    }:
        raise ValueError("rational solution is not exactly bound to the system")
    if claim["artifact_uri"] not in candidate["parents"]:
        raise ValueError("rational solution is missing system lineage")
    values = payload["values"]
    if not isinstance(values, list) or len(values) != columns:
        raise ValueError("rational solution is not a total vector")
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
    ):
        raise ValueError("rational solution producer identity is malformed")
    budget = payload["resource_budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"budget_version", "wall_seconds"}
        or budget["budget_version"] != "1"
        or type(budget["wall_seconds"]) is not int
        or not 1 <= budget["wall_seconds"] <= 60
    ):
        raise ValueError("rational solution resource budget is malformed")
    return [_rational(value) for value in values]


def _validate_inconsistency(
    payload: object,
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    rows: int,
    columns: int,
    variables: list[str],
) -> tuple[list[Fraction], Fraction]:
    if not isinstance(payload, dict) or set(payload) != {
        "certificate_schema_version",
        "system",
        "declared_scope",
        "left_witness",
        "rhs_pairing",
        "producer",
        "resource_budget",
        "method",
    }:
        raise ValueError("rational inconsistency certificate has an invalid shape")
    if (
        payload["certificate_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_SYSTEM"
        or payload["method"] != "DUAL_RREF_PAIRING_ONE"
    ):
        raise ValueError("rational inconsistency uses unsupported semantics")
    binding = payload["system"]
    if not isinstance(binding, dict) or set(binding) != _SYSTEM_BINDING_KEYS:
        raise ValueError("rational inconsistency system binding is malformed")
    expected_variable_digest = _sha256(_canonical_json({"variables": variables}))
    if binding != {
        "binding_version": "1",
        "system_artifact_uri": claim["artifact_uri"],
        "system_object_digest": claim["object_digest"],
        "system_payload_digest": claim["payload_digest"],
        "variable_order_digest": expected_variable_digest,
        "row_count": rows,
        "column_count": columns,
    }:
        raise ValueError("rational inconsistency is not exactly bound to the system")
    if claim["artifact_uri"] not in candidate["parents"]:
        raise ValueError("rational inconsistency is missing system lineage")
    values = payload["left_witness"]
    if not isinstance(values, list) or len(values) != rows:
        raise ValueError("left witness must contain one exact value per system row")
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
    ):
        raise ValueError("rational inconsistency producer identity is malformed")
    budget = payload["resource_budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"budget_version", "wall_seconds"}
        or budget["budget_version"] != "1"
        or type(budget["wall_seconds"]) is not int
        or not 1 <= budget["wall_seconds"] <= 60
    ):
        raise ValueError("rational inconsistency resource budget is malformed")
    pairing = _rational(payload["rhs_pairing"])
    if pairing != 1:
        raise ValueError("rational inconsistency witness is not normalized")
    return [_rational(value) for value in values], pairing


_LINEAR_REQUEST_KEYS = {
    "request_version",
    "claim",
    "candidate",
    "scope",
    "witness",
    "expected_bindings",
}
_LINEAR_WITNESS_ENVELOPE_KEYS = {
    "evidence_schema_version",
    "witness_format",
    "format_version",
    "role",
    "bindings",
    "payload",
}


def _check_linear_request_envelope(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != _LINEAR_REQUEST_KEYS:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _check_linear_artifacts(
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


def _check_linear_digests(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
    candidate_label: str,
) -> str | None:
    for artifact, label in (
        (claim, "linear-system"),
        (candidate, candidate_label),
        (witness, "linear witness"),
    ):
        if artifact["payload_digest"] != _sha256(_canonical_json(artifact["payload"])):
            return f"{label} payload digest does not match"
    return None


def _check_linear_binding_match(
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


def _check_linear_witness_envelope(
    envelope: object,
    expected_bindings: object,
    witness_format: str,
) -> str | None:
    if not isinstance(envelope, dict) or set(envelope) != _LINEAR_WITNESS_ENVELOPE_KEYS:
        return "linear witness envelope is malformed"
    if (
        envelope["evidence_schema_version"] != "1"
        or envelope["witness_format"] != witness_format
        or envelope["format_version"] != "1"
        or envelope["role"] != "SUPPORTS_CLAIM"
        or envelope["bindings"] != expected_bindings
    ):
        return "linear witness envelope is not exactly bound"
    return None


def _check_linear_witness_payload(
    witness: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
    candidate_key: str,
) -> str | None:
    if witness["payload"]["payload"] != {
        "system_uri": claim["artifact_uri"],
        candidate_key: candidate["artifact_uri"],
    }:
        return "linear witness points at different artifacts"
    if not {
        claim["artifact_uri"],
        candidate["artifact_uri"],
    }.issubset(set(witness["parents"])):
        return "linear witness is missing required lineage"
    return None


def _check_linear_witness(
    witness: dict[str, Any],
    expected_bindings: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness_format: str,
    candidate_key: str,
) -> str | None:
    error = _check_linear_witness_envelope(
        witness["payload"], expected_bindings, witness_format
    )
    if error is not None:
        return error
    return _check_linear_witness_payload(witness, claim, candidate, candidate_key)


def check_rational_solution(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a total exact vector satisfying every bound equation."""

    try:
        error = _check_linear_request_envelope(request)
        if error is not None:
            return _reject(error)
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        expected_bindings = request["expected_bindings"]
        error = _check_linear_artifacts(claim, candidate, witness, expected_bindings)
        if error is not None:
            return _reject(error)
        error = _check_linear_digests(claim, candidate, witness, "solution")
        if error is not None:
            return _reject(error)

        coefficients, rhs, variables = _validate_system(claim["payload"])
        values = _validate_solution(
            candidate["payload"],
            claim=claim,
            candidate=candidate,
            rows=len(coefficients),
            columns=len(variables),
            variables=variables,
        )
        error = _check_linear_binding_match(expected_bindings, claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_linear_witness(
            witness,
            expected_bindings,
            claim,
            candidate,
            "linear.rational_solution",
            "solution_uri",
        )
        if error is not None:
            return _reject(error)

        for row, expected in zip(coefficients, rhs, strict=True):
            if (
                sum(
                    (
                        coefficient * value
                        for coefficient, value in zip(row, values, strict=True)
                    ),
                    Fraction(0),
                )
                != expected
            ):
                return _reject("candidate does not satisfy every bound equation")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"replayed {len(coefficients)} equations over "
                f"{len(variables)} exact rational variables"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed checker request")


def check_rational_inconsistency(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a left witness proving the exact bound system inconsistent."""

    try:
        error = _check_linear_request_envelope(request)
        if error is not None:
            return _reject(error)
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        expected_bindings = request["expected_bindings"]
        error = _check_linear_artifacts(claim, candidate, witness, expected_bindings)
        if error is not None:
            return _reject(error)
        error = _check_linear_digests(claim, candidate, witness, "inconsistency")
        if error is not None:
            return _reject(error)

        coefficients, rhs, variables = _validate_system(claim["payload"])
        values, declared_pairing = _validate_inconsistency(
            candidate["payload"],
            claim=claim,
            candidate=candidate,
            rows=len(coefficients),
            columns=len(variables),
            variables=variables,
        )
        error = _check_linear_binding_match(expected_bindings, claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_linear_witness(
            witness,
            expected_bindings,
            claim,
            candidate,
            "linear.rational_inconsistency",
            "certificate_uri",
        )
        if error is not None:
            return _reject(error)

        for column in range(len(variables)):
            if (
                sum(
                    (
                        values[row] * coefficients[row][column]
                        for row in range(len(coefficients))
                    ),
                    Fraction(0),
                )
                != 0
            ):
                return _reject("left witness does not annihilate every column")
        actual_pairing = sum(
            (value * bound for value, bound in zip(values, rhs, strict=True)),
            Fraction(0),
        )
        if actual_pairing != declared_pairing or actual_pairing == 0:
            return _reject("left witness has no nonzero right-hand-side pairing")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "relation_id": "linear.relation.inconsistency-certificate-of",
            "relationship_source_artifact_uris": [candidate["artifact_uri"]],
            "relationship_target_artifact_uris": [claim["artifact_uri"]],
            "detail": (
                f"replayed a {len(values)}-entry left witness over "
                f"{len(variables)} exact rational columns"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed rational inconsistency request")
