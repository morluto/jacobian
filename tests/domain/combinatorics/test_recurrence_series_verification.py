from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.runtime.config import CheckerAuthorityMode

_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
_P_RECURSIVE_CONVENTION = "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"


@pytest.fixture
def combinatorics_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install combinatorics and its independent exact checkers only."""

    bundle = build_combinatorics_bundle()
    with open_domain_services(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        installed = DomainBundleInstaller(services.installation).install(
            PortfolioPlan(domain_bundles=(bundle,))
        )
        adapters, _ = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.application.verification,
            services.core.checkers,
            bundles={"combinatorics": (bundle, installed.installed["combinatorics"])},
            authorize=services.installation.authorizes_bundled_checkers,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
        yield services


_CASES = (
    (
        "combinatorics.recurrence.linear.evaluate",
        {
            "coefficients": [_q(1), _q(1)],
            "initial_values": [_q(0), _q(1)],
            "coefficient_convention": _RECURRENCE_CONVENTION,
            "scope": "PREFIX",
            "term_count": 8,
            "indices": [],
        },
    ),
    (
        "combinatorics.recurrence.p_recursive.evaluate",
        {
            "coefficient_polynomials": [[_q(1)], [_q(0), _q(-1)]],
            "initial_values": [_q(1)],
            "coefficient_convention": _P_RECURSIVE_CONVENTION,
            "polynomial_convention": "ASCENDING_POWERS_OF_N",
            "scope": "PREFIX",
            "term_count": 8,
            "indices": [],
        },
    ),
    (
        "combinatorics.generating_function.coefficients.compute",
        {
            "numerator": [_q(1)],
            "denominator": [_q(1), _q(-1), _q(-1)],
            "coefficient_convention": "ASCENDING_POWERS_OF_X",
            "expansion_point": "0",
            "truncation_order": 8,
        },
    ),
)


@pytest.mark.parametrize(("capability_id", "payload"), _CASES)
def test_recurrence_and_series_results_are_independently_verified(
    combinatorics_services,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    computed = combinatorics_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    verified = combinatorics_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(capability_id),
            input={
                "input": payload,
                "candidate": computed.output["result"],
            },
        )
    )
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == capability_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize(("capability_id", "payload"), _CASES)
def test_checker_rejects_contract_valid_false_results(
    combinatorics_services,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    runtime = combinatorics_services
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    forged_candidate = deepcopy(computed.output["result"])
    if capability_id == "combinatorics.recurrence.linear.evaluate":
        forged_candidate["replay_prefix"][7] = _q(14)
        forged_candidate["values"][7]["value"] = _q(14)
    elif capability_id == "combinatorics.recurrence.p_recursive.evaluate":
        forged_candidate["replay_prefix"][7] = _q(5039)
        forged_candidate["values"][7]["value"] = _q(5039)
    else:
        forged_candidate["coefficients"][7] = _q(22)
    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(capability_id),
            input={"input": payload, "candidate": forged_candidate},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_checker_runtime_binds_only_independent_source(
    combinatorics_services,
) -> None:
    descriptor = next(
        item
        for item in combinatorics_services.core.capabilities.catalog().capabilities
        if item.capability_id == "combinatorics.recurrence.linear.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {
        "jacobian.additive-combinatorics-checker-source",
        "jacobian.combinatorics-exact-checker-source",
    }


def test_checker_replays_a_result_above_python_default_integer_digit_limit(
    combinatorics_services,
) -> None:
    large = "9" * 64
    recurrence_input = {
        "coefficients": [{"num": large, "den": "1"}],
        "initial_values": [{"num": large, "den": "1"}],
        "coefficient_convention": _RECURRENCE_CONVENTION,
        "scope": "INDICES",
        "term_count": None,
        "indices": [68],
    }
    computed = combinatorics_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.evaluate",
            input=recurrence_input,
        )
    )
    verified = combinatorics_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.verify",
            input={
                "input": recurrence_input,
                "candidate": computed.output["result"],
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
