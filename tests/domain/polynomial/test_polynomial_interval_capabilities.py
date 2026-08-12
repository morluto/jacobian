from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_projection import project_operation_result
from jacobian.polynomial_interval_capabilities import (
    PolynomialIntervalEnclosureVerifyAdapter,
    install_polynomial_interval_capabilities,
)
from tests.support.capability_installations import install_capability_bundle
from tests.support.polynomials import univariate_term as _term


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


@pytest.fixture()
def installation(tmp_path: Path):
    with install_capability_bundle(
        tmp_path, install_polynomial_interval_capabilities
    ) as bundle:
        yield bundle


def test_verify_adapter_rejects_missing_authorized_checker(installation) -> None:
    adapters, _installed, _store = installation
    _enclose, verify = adapters
    assert verify is not None
    resources = replace(
        verify.resources,
        installation=replace(verify.resources.installation, checker_id=None),
    )

    with pytest.raises(RuntimeError, match="requires an authorized checker"):
        PolynomialIntervalEnclosureVerifyAdapter(resources)


def test_enclose_capability_computes_a_valid_bernstein_enclosure(
    installation,
) -> None:
    adapters, _installed, _store = installation
    enclose, _verify = adapters

    result = project_operation_result(
        enclose.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclose",
                input={
                    "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                    "interval": _interval("0", "1"),
                },
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["enclosure_kind"] == "BERNSTEIN_COEFFICIENT_BOUND"
    assert result.output["range_exactness"] == "ENCLOSURE_VALID_NOT_EXACT"
    assert result.output["degree"] == 1
    assert result.output["bernstein_coefficients"] == [
        {"num": "1", "den": "1"},
        {"num": "3", "den": "1"},
    ]
    assert result.output["lo"] == {"num": "1", "den": "1"}
    assert result.output["hi"] == {"num": "3", "den": "1"}
    assert result.output["polynomial_uri"] in result.artifact_uris
    assert result.output["enclosure_uri"] in result.artifact_uris


def test_enclose_capability_handles_a_quadratic_on_a_shifted_interval(
    installation,
) -> None:
    adapters, _installed, _store = installation
    enclose, _verify = adapters

    # p(x) = x^2 on [-1, 1]; Bernstein bound is [-1, 1] (valid, not exact range).
    result = project_operation_result(
        enclose.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclose",
                input={
                    "polynomial": _polynomial("x", [_term(1, 2)]),
                    "interval": _interval("-1", "1"),
                },
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["degree"] == 2
    assert result.output["bernstein_coefficients"] == [
        {"num": "1", "den": "1"},
        {"num": "-1", "den": "1"},
        {"num": "1", "den": "1"},
    ]
    assert result.output["lo"] == {"num": "-1", "den": "1"}
    assert result.output["hi"] == {"num": "1", "den": "1"}
    assert result.output["range_exactness"] == "ENCLOSURE_VALID_NOT_EXACT"


def test_verify_capability_confirms_a_valid_enclosure(installation) -> None:
    adapters, _installed, _store = installation
    enclose, verify = adapters
    assert verify is not None

    # First compute the enclosure, then verify the claimed values.
    enclose_result = project_operation_result(
        enclose.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclose",
                input={
                    "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                    "interval": _interval("0", "1"),
                },
            )
        )
    )
    claimed = enclose_result.output

    result = project_operation_result(
        verify.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclosure.verify",
                input={
                    "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                    "interval": _interval("0", "1"),
                    "claimed_bernstein_coefficients": claimed["bernstein_coefficients"],
                    "claimed_lo": claimed["lo"],
                    "claimed_hi": claimed["hi"],
                },
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["conclusion"] == "TRUE"
    assert result.output["checker_id"] is not None
    assert result.output["verification_record_uri"] is not None


def test_verify_capability_rejects_a_false_enclosure(installation) -> None:
    adapters, _installed, _store = installation
    _enclose, verify = adapters
    assert verify is not None

    # Claim Bernstein coefficients [0, 3] for p(x) = 2x + 1 on [0, 1]; the
    # independent replay computes [1, 3], so the checker must return FALSE.
    result = project_operation_result(
        verify.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclosure.verify",
                input={
                    "polynomial": _polynomial("x", [_term(2, 1), _term(1, 0)]),
                    "interval": _interval("0", "1"),
                    "claimed_bernstein_coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "3", "den": "1"},
                    ],
                    "claimed_lo": {"num": "0", "den": "1"},
                    "claimed_hi": {"num": "3", "den": "1"},
                },
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.verification_record_uri is not None
    assert result.output["conclusion"] == "FALSE"


def test_enclose_capability_rejects_a_degenerate_interval(installation) -> None:
    adapters, _installed, _store = installation
    enclose, _verify = adapters

    with pytest.raises(CapabilityInvocationError):
        enclose.invoke(
            CapabilityRequest(
                capability_id="polynomial.interval.enclose",
                input={
                    "polynomial": _polynomial("x", [_term(1, 0)]),
                    "interval": _interval("1", "1"),
                },
            )
        )
