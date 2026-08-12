from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.support.capability_installations import install_capability_bundle
from tests.support.polynomials import univariate_term as _term

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.polynomial_positivity_capabilities import (
    PolynomialIntervalPositivityVerifyAdapter,
    install_polynomial_positivity_capabilities,
)
from jacobian_checkers.polynomial_positivity import check_positivity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polynomial(variable: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variable": variable,
        "polynomial": {"terms": terms},
    }


def _interval(lo: str, hi: str) -> dict[str, Any]:
    return {
        "interval_schema_version": "1",
        "lo": {"num": lo, "den": "1"},
        "hi": {"num": hi, "den": "1"},
    }


# ---------------------------------------------------------------------------
# Checker unit tests
# ---------------------------------------------------------------------------


def _checker_request(
    *,
    polynomial: dict[str, Any],
    interval: dict[str, Any],
    positive: bool,
    sign_changes_lo: int,
    sign_changes_hi: int,
    roots_in_open: int,
    endpoint_root: bool,
    degree: int,
) -> dict[str, Any]:
    bindings = {
        "claim_digest": "sha256:" + "1" * 64,
        "semantics_digest": "sha256:" + "2" * 64,
        "candidate_digest": "sha256:" + "3" * 64,
        "scope_digest": "sha256:" + "4" * 64,
        "encoding_digest": None,
    }
    polynomial_uri = "artifact://sha256/" + "5" * 64
    decision_uri = "artifact://sha256/" + "6" * 64
    return {
        "request_version": "1",
        "claim": {
            "payload": {
                "claim_schema_version": "1",
                "predicate": "POLYNOMIAL_INTERVAL_STRICT_POSITIVITY",
                "domain": "QQ",
                "polynomial_uri": polynomial_uri,
                "interval": deepcopy(interval),
                "positive": positive,
            }
        },
        "scope": {
            "artifact_uri": polynomial_uri,
            "payload": polynomial,
        },
        "candidate": {
            "artifact_uri": decision_uri,
            "payload": {
                "decision_schema_version": "1",
                "polynomial_uri": polynomial_uri,
                "interval": deepcopy(interval),
                "degree": degree,
                "sturm_sequence": [],
                "sign_changes_at_lo": sign_changes_lo,
                "sign_changes_at_hi": sign_changes_hi,
                "roots_in_open_interval": roots_in_open,
                "endpoint_root": endpoint_root,
                "positive": positive,
                "backend": "sympy",
                "backend_version": "1.14.0",
            },
        },
        "certificate": {
            "payload": {
                "evidence_schema_version": "1",
                "certificate_type": ("polynomial.interval_sturm_positivity_replay"),
                "format_version": "1",
                "bindings": deepcopy(bindings),
                "payload_digest": "sha256:" + "0" * 64,
                "payload": {
                    "method": "STURM_SEQUENCE_REPLAY",
                    "polynomial_uri": polynomial_uri,
                    "interval": deepcopy(interval),
                    "degree": degree,
                    "sturm_sequence_length": max(1, degree + 1),
                    "sign_changes_at_lo": sign_changes_lo,
                    "sign_changes_at_hi": sign_changes_hi,
                    "roots_in_open_interval": roots_in_open,
                    "endpoint_root": endpoint_root,
                    "positive": positive,
                },
            }
        },
        "expected_bindings": deepcopy(bindings),
    }


def test_checker_accepts_positive_constant() -> None:
    # p(x) = 5 on [0, 1] — strictly positive, no roots.
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(5, 0)]),
            interval=_interval("0", "1"),
            positive=True,
            sign_changes_lo=0,
            sign_changes_hi=0,
            roots_in_open=0,
            endpoint_root=False,
            degree=0,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["relation_id"] == "polynomial.relation.valid-positivity-decision"


def test_checker_accepts_positive_linear() -> None:
    # p(x) = 2x + 1 on [0, 1] — p(0)=1>0, no roots in [0,1].
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(2, 1), _term(1, 0)]),
            interval=_interval("0", "1"),
            positive=True,
            sign_changes_lo=0,
            sign_changes_hi=0,
            roots_in_open=0,
            endpoint_root=False,
            degree=1,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_refutes_wrong_positivity() -> None:
    # p(x) = x - 1 on [0, 2] — p(1)=0, so NOT strictly positive.
    # Claim says positive=True, but the independent replay finds a root.
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(1, 1), _term(-1, 0)]),
            interval=_interval("0", "2"),
            positive=True,
            sign_changes_lo=1,
            sign_changes_hi=0,
            roots_in_open=1,
            endpoint_root=False,
            degree=1,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "FALSE"
    assert "relation_id" not in decision


def test_checker_accepts_correctly_refuted_positivity() -> None:
    # p(x) = x - 1 on [0, 2] — has a root at x=1, so positive=False is correct.
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(1, 1), _term(-1, 0)]),
            interval=_interval("0", "2"),
            positive=False,
            sign_changes_lo=1,
            sign_changes_hi=0,
            roots_in_open=1,
            endpoint_root=False,
            degree=1,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_accepts_quadratic_with_root_in_interval() -> None:
    # p(x) = x^2 - 1 on [-2, 2] — roots at x=-1 and x=1, not strictly positive.
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(1, 2), _term(-1, 0)]),
            interval=_interval("-2", "2"),
            positive=False,
            sign_changes_lo=2,
            sign_changes_hi=0,
            roots_in_open=2,
            endpoint_root=False,
            degree=2,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_accepts_positive_quadratic() -> None:
    # p(x) = x^2 + 1 on [0, 1] — always positive, no real roots.
    # Sturm sequence: [x^2+1, 2x, -1]. At 0: [1, 0, -1] → 1 sign change.
    # At 1: [2, 2, -1] → 1 sign change. V(0)-V(1) = 0 roots in (0,1].
    decision = check_positivity(
        _checker_request(
            polynomial=_polynomial("x", [_term(1, 2), _term(1, 0)]),
            interval=_interval("0", "1"),
            positive=True,
            sign_changes_lo=1,
            sign_changes_hi=1,
            roots_in_open=0,
            endpoint_root=False,
            degree=2,
        )
    )
    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_rejects_noncanonical_rational() -> None:
    request = _checker_request(
        polynomial=_polynomial("x", [_term(5, 0)]),
        interval=_interval("0", "1"),
        positive=True,
        sign_changes_lo=0,
        sign_changes_hi=0,
        roots_in_open=0,
        endpoint_root=False,
        degree=0,
    )
    request["candidate"]["payload"]["endpoint_root"] = "yes"

    decision = check_positivity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def installation(tmp_path: Path):
    with install_capability_bundle(
        tmp_path, install_polynomial_positivity_capabilities
    ) as bundle:
        yield bundle


def test_verify_adapter_rejects_missing_authorized_checker(installation) -> None:
    adapters, _installed, _store = installation
    _decide, verify = adapters
    assert verify is not None
    resources = replace(
        verify.resources,
        installation=replace(verify.resources.installation, checker_id=None),
    )

    with pytest.raises(RuntimeError, match="requires an authorized checker"):
        PolynomialIntervalPositivityVerifyAdapter(resources)


def test_decide_capability_finds_positive_linear(installation) -> None:
    adapters, _installed, _store = installation
    decide, _verify = adapters

    result = decide.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.decide",
            input={
                "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                "interval": _interval("0", "1"),
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["positive"] is True
    assert result.output["roots_in_open_interval"] == 0
    assert result.output["endpoint_root"] is False


def test_decide_capability_detects_root_in_interval(installation) -> None:
    adapters, _installed, _store = installation
    decide, _verify = adapters

    # p(x) = x - 1 on [0, 2] — root at x=1, not strictly positive.
    result = decide.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.decide",
            input={
                "polynomial": _polynomial("x", [_term(1, 1), _term(-1, 0)]),
                "interval": _interval("0", "2"),
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["positive"] is False
    assert result.output["roots_in_open_interval"] == 1
    assert result.output["sign_changes_at_lo"] == 1
    assert result.output["sign_changes_at_hi"] == 0


def test_decide_capability_detects_endpoint_root(installation) -> None:
    adapters, _installed, _store = installation
    decide, _verify = adapters

    # p(x) = x on [0, 1] — root at x=0 (the left endpoint), not strictly positive.
    result = decide.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.decide",
            input={
                "polynomial": _polynomial("x", [_term(1, 1)]),
                "interval": _interval("0", "1"),
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["positive"] is False
    assert result.output["endpoint_root"] is True


def test_verify_capability_confirms_positive_decision(installation) -> None:
    adapters, _installed, _store = installation
    decide, verify = adapters
    assert verify is not None

    decide_result = decide.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.decide",
            input={
                "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                "interval": _interval("0", "1"),
            },
        )
    )
    claimed = decide_result.output

    result = verify.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.verify",
            input={
                "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                "interval": _interval("0", "1"),
                "claimed_positive": claimed["positive"],
                "claimed_sign_changes_at_lo": claimed["sign_changes_at_lo"],
                "claimed_sign_changes_at_hi": claimed["sign_changes_at_hi"],
                "claimed_roots_in_open_interval": claimed["roots_in_open_interval"],
                "claimed_endpoint_root": claimed["endpoint_root"],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["conclusion"] == "TRUE"
    assert result.output["checker_id"] is not None
    assert result.output["verification_record_uri"] is not None


def test_verify_capability_refutes_false_positive_claim(installation) -> None:
    adapters, _installed, _store = installation
    _decide, verify = adapters
    assert verify is not None

    # p(x) = x - 1 on [0, 2] has a root at x=1. Claim positive=True with
    # inconsistent sign-change counts — the checker independently finds the
    # root and returns FALSE.
    result = verify.invoke(
        CapabilityRequest(
            capability_id="polynomial.interval.positivity.verify",
            input={
                "polynomial": _polynomial("x", [_term(1, 1), _term(-1, 0)]),
                "interval": _interval("0", "2"),
                "claimed_positive": True,
                "claimed_sign_changes_at_lo": 0,
                "claimed_sign_changes_at_hi": 0,
                "claimed_roots_in_open_interval": 0,
                "claimed_endpoint_root": False,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.verification_record_uri is not None
    assert result.output["conclusion"] == "FALSE"
