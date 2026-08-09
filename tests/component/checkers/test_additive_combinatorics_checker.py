from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest

from jacobian_checkers.additive_combinatorics import (
    check_cyclic_difference_set_extension,
    check_cyclic_perfect_difference_set,
    check_integer_sidon,
)

_BASE = [1, 2, 4, 8, 13]
_META_INTEGER = {
    "exactness": "EXACT_INTEGER",
    "determinism": "DETERMINISTIC",
    "backend": "python-stdlib",
    "verification": "UNVERIFIED",
}
_META_FINITE = {
    "exactness": "EXACT_FINITE",
    "determinism": "DETERMINISTIC",
    "backend": "python-stdlib",
    "verification": "UNVERIFIED",
}


def _artifact(
    character: str,
    payload: dict[str, Any],
    *,
    semantics: str,
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(character),
        "object_digest": "sha256:" + character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(chr(ord(character) + 1)),
        "semantics_uri": semantics,
        "parents": parents,
        "payload": payload,
    }


def _request(
    operation_id: str,
    witness_format: str,
    source: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    semantics = _uri("e")
    semantics_artifact = _artifact(
        "e", {"kind": "semantics"}, semantics=_uri("0"), parents=[]
    )
    semantics_artifact["object_digest"] = "sha256:" + "8" * 64
    claim = _artifact("1", source, semantics=semantics, parents=[])
    candidate = _artifact(
        "3", result, semantics=semantics, parents=[claim["artifact_uri"]]
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": semantics_artifact["object_digest"],
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    witness = _artifact(
        "5",
        {
            "evidence_schema_version": "1",
            "witness_format": witness_format,
            "format_version": "1",
            "role": "SUPPORTS_CLAIM",
            "bindings": bindings,
            "payload": {
                "operation_id": operation_id,
                "input_uri": claim["artifact_uri"],
                "result_uri": candidate["artifact_uri"],
            },
        },
        semantics=semantics,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics_artifact,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


def _sidon_case() -> dict[str, Any]:
    differences = [
        {
            "minuend": str(left),
            "subtrahend": str(right),
            "difference": str(left - right),
        }
        for left in _BASE
        for right in _BASE
        if left != right
    ]
    return _request(
        "combinatorics.integer_set.sidon.decide",
        "combinatorics.integer-sidon.ordered-difference-replay",
        {"elements": [str(value) for value in _BASE]},
        {
            "semantics_version": "integer-sidon.ordered-differences.v1",
            "normalized_elements": [str(value) for value in _BASE],
            "ordered_differences": differences,
            "is_sidon": True,
            **_META_INTEGER,
        },
    )


def _perfect_case() -> dict[str, Any]:
    return _request(
        "combinatorics.cyclic_difference_set.perfect.decide",
        "combinatorics.cyclic-pds.residue-profile-replay",
        {"modulus": 7, "residues": [0, 1, 3]},
        {
            "semantics_version": "cyclic-perfect-difference-set.v1",
            "modulus": 7,
            "normalized_residues": [0, 1, 3],
            "order": 3,
            "expected_modulus": 7,
            "difference_multiplicities": [
                {"residue": residue, "multiplicity": 1} for residue in range(1, 7)
            ],
            "missing_residues": [],
            "repeated_residues": [],
            "is_perfect": True,
            **_META_FINITE,
        },
    )


def _negative_extension_case() -> dict[str, Any]:
    return _request(
        "combinatorics.cyclic_difference_set.extension.decide",
        "combinatorics.cyclic-pds-extension.exhaustive-replay",
        {"base_elements": [str(value) for value in _BASE], "target_order": 6},
        {
            "semantics_version": "cyclic-pds-extension.fixed-order.v1",
            "target_order": 6,
            "modulus": 31,
            "base_residues": _BASE,
            "candidate_space_size": 26,
            "decision": "DOES_NOT_EXTEND",
            "extension": [],
            "coverage": "ALL_CANDIDATES",
            **_META_FINITE,
        },
    )


@pytest.mark.parametrize(
    ("checker", "case_request"),
    (
        (check_integer_sidon, _sidon_case()),
        (check_cyclic_perfect_difference_set, _perfect_case()),
        (check_cyclic_difference_set_extension, _negative_extension_case()),
    ),
)
def test_additive_combinatorics_checkers_accept_complete_exact_replay(
    checker: Any,
    case_request: dict[str, Any],
) -> None:
    checked = checker(case_request)
    assert checked["accepted"] is True
    assert checked["conclusion"] == "TRUE"


def test_sidon_checker_rejects_a_missing_difference_with_fresh_digest() -> None:
    forged = copy.deepcopy(_sidon_case())
    forged["candidate"]["payload"]["ordered_differences"].pop()
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    checked = check_integer_sidon(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_perfect_checker_rejects_a_false_multiplicity_with_fresh_digest() -> None:
    forged = copy.deepcopy(_perfect_case())
    forged["candidate"]["payload"]["difference_multiplicities"][0]["multiplicity"] = 2
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    assert check_cyclic_perfect_difference_set(forged)["accepted"] is False


def test_extension_checker_rejects_wrong_scope_with_fresh_digest() -> None:
    forged = copy.deepcopy(_negative_extension_case())
    forged["candidate"]["payload"]["target_order"] = 5
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    assert check_cyclic_difference_set_extension(forged)["accepted"] is False
