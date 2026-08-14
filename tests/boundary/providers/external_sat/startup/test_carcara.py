from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import (
    OperationRequest,
    ProviderAvailability,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.smt import SmtResourceBudget
from jacobian.providers.external_solver_runtime import (
    carcara_provider_runtime,
    cvc5_provider_runtime,
)
from jacobian.sat_smt.cvc5 import bind_cvc5_operation
from jacobian.sat_smt.smt_operations import install_smt_unsat_proof_checker

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def carcara_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    carcara = carcara_provider_runtime()
    if carcara.availability is not ProviderAvailability.AVAILABLE:
        pytest.skip("the exact operator-provenanced Carcara runtime is unavailable")
    cvc5 = cvc5_provider_runtime()
    if cvc5.availability is not ProviderAvailability.AVAILABLE:
        pytest.skip("the pinned cvc5 runtime is unavailable")
    with open_domain_services(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        services.installation.register_operation(
            bind_cvc5_operation(services.core.smt, cvc5)
        )
        verifier, installation = install_smt_unsat_proof_checker(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.core.smt,
            services.verification,
            services.core.checkers,
            carcara,
            authorize_checker=services.installation.authorize_bundled_checkers,
        )
        assert verifier is not None
        assert installation.checker_id is not None
        services.installation.register_operation(verifier)
        yield services


def _produce(runtime: DomainTestServices, logic: str, fixture: str):
    return runtime.core.operations.invoke(
        OperationRequest(
            operation_id="smt.unsat_proof.find",
            input={
                "logic": logic,
                "smtlib_text": (_FIXTURES / fixture).read_text(encoding="ascii"),
                "resource_budget": {"wall_seconds": 5},
            },
        )
    )


def _verify(runtime: DomainTestServices, proof_uri: str):
    return runtime.core.operations.invoke(
        OperationRequest(
            operation_id="smt.unsat_proof.verify",
            input={"proof_uri": proof_uri},
        )
    )


def test_zero_hole_qf_uf_proof_is_independently_verified(
    carcara_services: DomainTestServices,
) -> None:
    produced = _produce(carcara_services, "QF_UF", "qf_uf_equality_unsat.smt2")

    assert produced.output["contains_holes"] is False
    assert "conclusion" not in produced.output
    verified = _verify(carcara_services, produced.output["proof_uri"])

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED_UNSAT"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["verification_record_uri"] is not None


@pytest.mark.parametrize(
    ("logic", "fixture"),
    (
        ("QF_LIA", "qf_lia_bounds_unsat.smt2"),
        ("QF_LRA", "qf_lra_bounds_unsat.smt2"),
    ),
)
def test_holey_arithmetic_proofs_remain_unverified(
    carcara_services: DomainTestServices,
    logic: str,
    fixture: str,
) -> None:
    produced = _produce(carcara_services, logic, fixture)

    assert produced.output["contains_holes"] is True
    assert produced.output["alethe_hole_count"] >= 1
    checked = _verify(carcara_services, produced.output["proof_uri"])

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["verification_record_uri"] is None


def test_unknown_rule_is_not_silently_treated_as_verified(
    carcara_services: DomainTestServices,
) -> None:
    produced = _produce(carcara_services, "QF_UF", "qf_uf_equality_unsat.smt2")
    resolved = carcara_services.core.smt.resolve_proof(produced.output["proof_uri"])
    unknown_rule = resolved.proof.raw_bytes().replace(
        b":rule resolution",
        b":rule jacobian_unknown_rule",
    )
    mutated = carcara_services.core.smt.put_proof(
        problem_uri=produced.output["problem_uri"],
        proof=unknown_rule,
        producer=resolved.proof.producer,
        resource_budget=SmtResourceBudget(wall_seconds=5),
    )

    checked = _verify(carcara_services, mutated.artifact_uri)

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["verification_record_uri"] is None
