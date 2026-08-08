from __future__ import annotations

from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.results import ExecutionStatus
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.runtime.model import JacobianRuntime


def _poly(*coefficients_ascending: int) -> dict[str, object]:
    return {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": _q(coefficient), "exponents": [exponent]}
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


def _poly_xy(*terms: tuple[tuple[int, int], int]) -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(coefficient),
                    "exponents": list(exponents),
                }
                for exponents, coefficient in terms
            ]
        },
    }


def _install_verification(
    fresh_complete_runtime: JacobianRuntime, *, authorize: bool
) -> tuple[object, ...]:
    portfolio = build_builtin_portfolio()
    bundles = {
        domain_id: (
            portfolio.bundle_for(domain_id),
            fresh_complete_runtime.portfolio.domain_bundles[domain_id],
        )
        for domain_id in ("polynomial", "matrix", "probability")
    }
    adapters, _ = install_exact_domain_verification(
        fresh_complete_runtime.core.store,
        fresh_complete_runtime.core.schemas,
        fresh_complete_runtime.core.artifacts,
        fresh_complete_runtime.services.verification,
        fresh_complete_runtime.core.checkers,
        bundles=bundles,
        authorize=authorize,
    )
    for adapter in adapters:
        fresh_complete_runtime.core.capabilities.register(adapter)
    return adapters


def _gcd_input() -> dict[str, object]:
    return {
        "left": _poly(-1, 0, 1),
        "right": _poly(0, 1, 1),
    }


def _computed_gcd(fresh_complete_runtime: JacobianRuntime):
    return fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.gcd",
            input=_gcd_input(),
        )
    )


def test_public_seam_verifies_exact_producer_result(fresh_complete_runtime) -> None:
    adapters = _install_verification(fresh_complete_runtime, authorize=True)
    gcd_adapter = next(
        adapter
        for adapter in adapters
        if adapter.descriptor.capability_id == "polynomial.gcd.verify"
    )
    provider_runtime = gcd_adapter.descriptor.provider_runtime
    assert provider_runtime is not None
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}
    computed = _computed_gcd(fresh_complete_runtime)

    verified = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.gcd.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": _gcd_input(),
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "polynomial.compute.gcd"
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    record = fresh_complete_runtime.core.store.get(
        verified.output["verification_record_uri"]
    )
    parsed = InlineExactVerificationRecord.model_validate(record.payload)
    assert verified.artifact_uris == (
        verified.output["verification_record_uri"],
        parsed.semantics_uri,
    )


def test_public_seam_rejects_validly_shaped_false_result(
    fresh_complete_runtime,
) -> None:
    _install_verification(fresh_complete_runtime, authorize=True)
    _computed_gcd(fresh_complete_runtime)

    false_candidate = {
        "gcd": _poly(1),
        "bezout": {
            "left_multiplier": _poly(),
            "right_multiplier": _poly(),
        },
        "normalization": "MONIC",
    }

    rejected = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.gcd.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": _gcd_input(),
                "candidate": false_candidate,
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_public_seam_reports_valid_multivariate_result_as_unsupported(
    fresh_complete_runtime,
) -> None:
    _install_verification(fresh_complete_runtime, authorize=True)
    resultant_input = {
        "left": _poly_xy(((1, 0), 1), ((0, 1), 1)),
        "right": _poly_xy(((1, 0), 1), ((0, 0), 1)),
        "elimination_variable": "x",
    }
    computed = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.resultant",
            input=resultant_input,
        )
    )

    checked = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.resultant.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "input": resultant_input,
                "candidate": computed.output["result"],
            },
        )
    )

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "UNSUPPORTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["witness_uri"] is None
    assert checked.output["verification_record_uri"] is None
    assert checked.assurance.level is CapabilityAssuranceLevel.COMPUTED
