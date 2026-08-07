from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.smt import SmtResourceBudget
from jacobian.providers.external_solver_runtime import carcara_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime

_FIXTURES = (
    Path(__file__).resolve().parents[5]
    / "tests"
    / "boundary"
    / "providers"
    / "external_sat"
    / "fixtures"
)


@pytest.fixture(scope="module")
def carcara_runtime(
    tmp_path_factory: pytest.TempPathFactory,
    authorized_portfolio_template: Path,
) -> JacobianRuntime:
    runtime = carcara_provider_runtime()
    if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
        pytest.skip("the exact operator-provenanced Carcara runtime is unavailable")
    root = tmp_path_factory.mktemp("carcara-runtime")
    shutil.copytree(authorized_portfolio_template, root, dirs_exist_ok=True)
    installed = create_runtime(
        root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert installed.portfolio.carcara_runtime == runtime
    try:
        yield installed
    finally:
        installed.close()


def _produce(runtime: JacobianRuntime, logic: str, fixture: str):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="smt.unsat_proof.find",
            input={
                "logic": logic,
                "smtlib_text": (_FIXTURES / fixture).read_text(encoding="ascii"),
                "resource_budget": {"wall_seconds": 5},
            },
        )
    )


def _verify(runtime: JacobianRuntime, proof_uri: str):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="smt.unsat_proof.verify",
            mode=CapabilityMode.VERIFY,
            input={"proof_uri": proof_uri},
        )
    )


def test_zero_hole_qf_uf_proof_is_independently_verified(
    carcara_runtime: JacobianRuntime,
) -> None:
    produced = _produce(carcara_runtime, "QF_UF", "qf_uf_equality_unsat.smt2")

    assert produced.output["contains_holes"] is False
    assert produced.output["conclusion"] == "UNKNOWN"
    verified = _verify(carcara_runtime, produced.output["proof_uri"])

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED_UNSAT"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] is not None


@pytest.mark.parametrize(
    ("logic", "fixture"),
    (
        ("QF_LIA", "qf_lia_bounds_unsat.smt2"),
        ("QF_LRA", "qf_lra_bounds_unsat.smt2"),
    ),
)
def test_holey_arithmetic_proofs_remain_unverified(
    carcara_runtime: JacobianRuntime,
    logic: str,
    fixture: str,
) -> None:
    produced = _produce(carcara_runtime, logic, fixture)

    assert produced.output["contains_holes"] is True
    assert produced.output["alethe_hole_count"] >= 1
    checked = _verify(carcara_runtime, produced.output["proof_uri"])

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert checked.output["verification_record_uri"] is None


def test_unknown_rule_is_not_silently_treated_as_verified(
    carcara_runtime: JacobianRuntime,
) -> None:
    produced = _produce(carcara_runtime, "QF_UF", "qf_uf_equality_unsat.smt2")
    resolved = carcara_runtime.core.smt.resolve_proof(produced.output["proof_uri"])
    unknown_rule = resolved.proof.raw_bytes().replace(
        b":rule resolution",
        b":rule jacobian_unknown_rule",
    )
    mutated = carcara_runtime.core.smt.put_proof(
        problem_uri=produced.output["problem_uri"],
        proof=unknown_rule,
        producer=carcara_runtime.portfolio.cvc5_runtime,
        resource_budget=SmtResourceBudget(wall_seconds=5),
    )

    checked = _verify(carcara_runtime, mutated.artifact_uri)

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["verification_record_uri"] is None
