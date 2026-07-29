from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from typing import Any

import pytest
from tests.helpers.artifacts import artifact_uri as _uri
from tests.helpers.artifacts import canonical_digest as _digest

import jacobian_checkers.finite_posets as checker_module
from jacobian.contracts.posets import (
    FinitePosetRequest,
    LinearExtensionRequest,
    MobiusFunctionRequest,
    PosetRequest,
)
from jacobian.domains.posets.operations import (
    _linear_extensions,
    _materialize,
    _mobius,
    _width,
)
from jacobian_checkers.finite_posets import (
    check_finite_poset_materialization,
    check_linear_extension_count,
    check_poset_mobius_function,
    check_poset_width,
)


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


_PRESENTATION = FinitePosetRequest(
    elements=("0", "a", "b", "1"),
    relation=(
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ),
    interpretation="COVER_EDGES",
)
_MATERIALIZED = _materialize(_PRESENTATION).value
_POSET = _MATERIALIZED.poset
_WIDTH = PosetRequest(poset=_POSET)
_LINEAR = LinearExtensionRequest(poset=_POSET)
_MOBIUS = MobiusFunctionRequest(poset=_POSET)

_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_finite_poset_materialization,
        _request(
            "poset.finite.materialize",
            "poset.finite.closure-reduction-replay",
            _PRESENTATION.model_dump(mode="json"),
            _MATERIALIZED.model_dump(mode="json"),
        ),
    ),
    (
        check_poset_width,
        _request(
            "poset.width.compute",
            "poset.width.dilworth-dual-replay",
            _WIDTH.model_dump(mode="json"),
            _width(_WIDTH).value.model_dump(mode="json"),
        ),
    ),
    (
        check_linear_extension_count,
        _request(
            "poset.linear_extensions.count",
            "poset.linear-extensions.complete-ideal-dp-replay",
            _LINEAR.model_dump(mode="json"),
            _linear_extensions(_LINEAR).value.model_dump(mode="json"),
        ),
    ),
    (
        check_poset_mobius_function,
        _request(
            "poset.mobius_function.compute",
            "poset.mobius.interval-convolution-replay",
            _MOBIUS.model_dump(mode="json"),
            _mobius(_MOBIUS).value.model_dump(mode="json"),
        ),
    ),
)


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_poset_checkers_accept_complete_exact_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    checked = checker(case_request)
    assert checked["accepted"] is True
    assert checked["conclusion"] == "TRUE"


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_poset_checkers_reject_payload_substitution_with_fresh_digest(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    forged = copy.deepcopy(case_request)
    candidate = forged["candidate"]
    candidate["payload"]["backend_version"] = "forged"
    candidate["payload_digest"] = _digest(candidate["payload"])
    assert checker(forged)["accepted"] is False


def test_width_checker_rejects_comparable_reported_antichain() -> None:
    forged = copy.deepcopy(_CASES[1][1])
    candidate = forged["candidate"]
    candidate["payload"]["maximum_antichain"] = ["0", "1"]
    candidate["payload_digest"] = _digest(candidate["payload"])
    assert check_poset_width(forged)["accepted"] is False


def test_linear_checker_rejects_a_self_consistent_wrong_top_count() -> None:
    forged = copy.deepcopy(_CASES[2][1])
    candidate = forged["candidate"]
    candidate["payload"]["count"] = 3
    candidate["payload"]["states"][-1]["count"] = 3
    candidate["payload"]["memo_digest"] = _digest(candidate["payload"]["states"])
    candidate["payload_digest"] = _digest(candidate["payload"])
    assert check_linear_extension_count(forged)["accepted"] is False


def test_mobius_checker_rejects_changed_value_and_contribution() -> None:
    forged = copy.deepcopy(_CASES[3][1])
    candidate = forged["candidate"]
    target = next(
        item
        for item in candidate["payload"]["values"]
        if item["lower"] == "0" and item["upper"] == "1"
    )
    target["value"] = 0
    target["recurrence_contributions"] = []
    candidate["payload_digest"] = _digest(candidate["payload"])
    assert check_poset_mobius_function(forged)["accepted"] is False


def test_checker_source_does_not_import_networkx_producer_or_contracts() -> None:
    source = inspect.getsource(checker_module)
    assert "import networkx" not in source
    assert "jacobian.domains.posets" not in source
    assert "jacobian.contracts.posets" not in source
