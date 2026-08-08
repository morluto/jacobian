from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest

import jacobian_checkers.simplicial_topology as checker_module
from jacobian.contracts.topology import (
    ChainCoefficientRing,
    ChainComplexRequest,
    HomologyConvention,
    IntegralSimplicialHomologyRequest,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)
from jacobian.domains.topology.operations import (
    _canonicalize,
    _chain_result,
    _homology,
    _integral_homology,
)
from jacobian_checkers.simplicial_topology import (
    check_integral_simplicial_homology,
    check_simplicial_chain_complex,
    check_simplicial_complex_canonicalization,
    check_simplicial_homology,
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


_PRESENTATION = {
    "vertices": ["c", "a", "b"],
    "facets": [["b", "a"], ["c", "b"], ["a", "c"]],
}
_CANONICALIZATION_REQUEST = SimplicialComplexRequest.model_validate(_PRESENTATION)
_CANONICALIZATION_RESULT = _canonicalize(_CANONICALIZATION_REQUEST).value
_COMPLEX = _CANONICALIZATION_RESULT.complex
_CHAIN_REQUEST = ChainComplexRequest(
    complex=_COMPLEX,
    coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
    prime=2,
    convention=HomologyConvention.UNREDUCED,
)
_INTEGER_CHAIN_REQUEST = ChainComplexRequest(
    complex=_COMPLEX,
    coefficient_ring=ChainCoefficientRing.INTEGER,
    convention=HomologyConvention.UNREDUCED,
)
_HOMOLOGY_REQUEST = SimplicialHomologyRequest(
    complex=_COMPLEX,
    prime=2,
    convention=HomologyConvention.UNREDUCED,
)
_REDUCED_HOMOLOGY_REQUEST = SimplicialHomologyRequest(
    complex=_COMPLEX,
    prime=3,
    convention=HomologyConvention.REDUCED,
)
_INTEGRAL_HOMOLOGY_REQUEST = IntegralSimplicialHomologyRequest(
    complex=_COMPLEX,
    convention=HomologyConvention.UNREDUCED,
)
_HOMOLOGY_CASE = _request(
    "topology.simplicial_homology.compute",
    "topology.simplicial-homology.modular-replay",
    _HOMOLOGY_REQUEST.model_dump(mode="json"),
    _homology(_HOMOLOGY_REQUEST).value.model_dump(mode="json"),
)

_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_simplicial_complex_canonicalization,
        _request(
            "topology.simplicial_complex.canonicalize",
            "topology.simplicial-complex.closure-replay",
            _CANONICALIZATION_REQUEST.model_dump(mode="json"),
            _CANONICALIZATION_RESULT.model_dump(mode="json"),
        ),
    ),
    (
        check_simplicial_chain_complex,
        _request(
            "topology.simplicial_complex.chain_complex.compute",
            "topology.simplicial-chain.boundary-replay",
            _CHAIN_REQUEST.model_dump(mode="json"),
            _chain_result(_CHAIN_REQUEST).model_dump(mode="json"),
        ),
    ),
    (
        check_simplicial_chain_complex,
        _request(
            "topology.simplicial_complex.chain_complex.compute",
            "topology.simplicial-chain.boundary-replay",
            _INTEGER_CHAIN_REQUEST.model_dump(mode="json"),
            _chain_result(_INTEGER_CHAIN_REQUEST).model_dump(mode="json"),
        ),
    ),
    (
        check_simplicial_homology,
        _HOMOLOGY_CASE,
    ),
    (
        check_simplicial_homology,
        _request(
            "topology.simplicial_homology.compute",
            "topology.simplicial-homology.modular-replay",
            _REDUCED_HOMOLOGY_REQUEST.model_dump(mode="json"),
            _homology(_REDUCED_HOMOLOGY_REQUEST).value.model_dump(mode="json"),
        ),
    ),
    (
        check_integral_simplicial_homology,
        _request(
            "topology.simplicial_homology.integral.compute",
            "topology.simplicial-homology.integral-smith-certificate-v1",
            _INTEGRAL_HOMOLOGY_REQUEST.model_dump(mode="json"),
            _integral_homology(_INTEGRAL_HOMOLOGY_REQUEST).value.model_dump(
                mode="json"
            ),
        ),
    ),
)


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_topology_checkers_accept_complete_exact_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    checked = checker(case_request)

    assert checked["accepted"] is True
    assert checked["conclusion"] == "TRUE"
    assert checked["arithmetic"] == "EXACT_INTEGER"


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_topology_checkers_reject_payload_substitution_with_fresh_digest(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    forged = copy.deepcopy(case_request)
    candidate = forged["candidate"]
    candidate["payload"]["backend_version"] = "forged"
    candidate["payload_digest"] = _digest(candidate["payload"])

    checked = checker(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_homology_checker_rejects_a_noncycle_representative() -> None:
    forged = copy.deepcopy(_HOMOLOGY_CASE)
    candidate = forged["candidate"]
    candidate["payload"]["groups"][1]["cycle_basis"][0]["coefficients"] = [1, 0, 0]
    candidate["payload"]["groups"][1]["homology_basis"][0]["coefficients"] = [
        1,
        0,
        0,
    ]
    candidate["payload_digest"] = _digest(candidate["payload"])

    checked = check_simplicial_homology(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_integral_homology_checker_rejects_a_forged_unimodular_transform() -> None:
    case = next(
        request
        for checker, request in _CASES
        if checker is check_integral_simplicial_homology
    )
    forged = copy.deepcopy(case)
    candidate = forged["candidate"]
    transformation = candidate["payload"]["groups"][1]["outgoing_smith_certificate"][
        "right_transformation"
    ]["entries"]
    transformation[0][0] = str(int(transformation[0][0]) + 1)
    candidate["payload_digest"] = _digest(candidate["payload"])

    checked = check_integral_simplicial_homology(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_prime_field_boundary_rejects_a_missing_prime() -> None:
    with pytest.raises(ValueError, match="prime-field boundary requires a prime"):
        checker_module._boundary(
            _COMPLEX.model_dump(mode="json"),
            1,
            ring="PRIME_FIELD",
            prime=None,
        )


def test_chain_checker_rejects_an_incomplete_reconstructed_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        request
        for checker, request in _CASES
        if checker is check_simplicial_chain_complex
    )
    original_boundary = checker_module._boundary

    def incomplete_boundary(*args: Any, **kwargs: Any) -> Any:
        dimension = args[1]
        if dimension == 0:
            return None
        return original_boundary(*args, **kwargs)

    monkeypatch.setattr(checker_module, "_boundary", incomplete_boundary)

    checked = check_simplicial_chain_complex(copy.deepcopy(case))

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_checker_source_does_not_import_topology_producer_or_contracts() -> None:
    source = inspect.getsource(checker_module)

    assert "jacobian.domains.topology" not in source
    assert "jacobian.contracts.topology" not in source
