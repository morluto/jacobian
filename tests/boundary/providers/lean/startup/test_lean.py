from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from tests.support.provider_lean import (
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    pinned_mathlib_runtime_available,
)

from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.lean import (
    LeanDeclarationSearchRequest,
    LeanEnvironment,
)
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    Coverage,
    InputStatus,
    Method,
    Verification,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanSubprocessDeclarationBackend,
)
from jacobian.runtime import CheckerAuthorityMode, create_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MATHLIB_OLEAN = (
    PROJECT_ROOT
    / "lean"
    / ".lake"
    / "packages"
    / "mathlib"
    / ".lake"
    / "build"
    / "lib"
    / "lean"
    / "Mathlib.olean"
)

pytestmark = [
    pytest.mark.skipif(
        not pinned_mathlib_runtime_available(),
        reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    ),
]


def test_core_declaration_catalog_matches_a_fresh_scan_and_detects_tampering(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> None:
    indexed_root = tmp_path / "indexed"
    fresh_root = tmp_path / "fresh"
    shutil.copytree(authorized_portfolio_template, indexed_root)
    shutil.copytree(authorized_portfolio_template, fresh_root)
    indexed_runtime = create_runtime(
        indexed_root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    fresh_runtime = create_runtime(
        fresh_root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    indexed = indexed_runtime.portfolio.lean_declarations
    fresh = fresh_runtime.portfolio.lean_declarations
    assert indexed is not None
    assert fresh is not None
    indexed_backend = indexed.backend
    fresh_backend = fresh.backend
    assert isinstance(indexed_backend, LeanSubprocessDeclarationBackend)
    assert isinstance(fresh_backend, LeanSubprocessDeclarationBackend)
    seed = LeanDeclarationSearchRequest(
        environment=LeanEnvironment.CORE,
        name_contains="Nat.add",
        result_limit=2,
    )
    target = LeanDeclarationSearchRequest(
        environment=LeanEnvironment.CORE,
        name_contains="Nat.mul",
        result_limit=2,
    )
    try:
        indexed.search(seed)
        reused = indexed.search(target)
        baseline = fresh.search(target)

        assert reused.declarations == baseline.declarations
        assert reused.scanned_declarations == baseline.scanned_declarations
        assert reused.stop_reason is baseline.stop_reason

        entry = indexed_backend._sessions[LeanEnvironment.CORE]
        index_path = cast(Any, entry.session)._index_path
        index_path.write_text("tampered\n", encoding="utf-8")

        with pytest.raises(LeanDeclarationBackendError) as raised:
            indexed.search(target)

        assert raised.value.code == "LEAN_QUERY_INDEX_CHANGED"
        assert LeanEnvironment.CORE not in indexed_backend._sessions
    finally:
        indexed_backend.close()
        fresh_backend.close()


def test_core_dependency_graph_is_bounded_and_materialized(tmp_path: Path) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.dependencies",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "root_declaration": "Nat.add_comm",
                "max_depth": 1,
                "max_nodes": 40,
            },
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.assurance.verification_record_uri is None
    assert result.output["nodes"][0] == {
        "name": "Nat.add_comm",
        "kind": "THEOREM",
        "depth": 0,
    }
    assert len(result.output["nodes"]) <= 40
    assert result.output["dependency_graph_uri"] in result.artifact_uris
    artifact = runtime.core.store.get(result.output["dependency_graph_uri"])
    assert artifact.payload["nodes"][0]["name"] == "Nat.add_comm"
    assert artifact.payload["query"]["max_depth"] == 1


@pytest.mark.skipif(
    not MATHLIB_OLEAN.is_file(),
    reason="the pinned mathlib runtime has not been built",
)
@pytest.mark.timeout(240)
def test_mathlib_discovery_composes_with_bound_sqrt_two_verification(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None
    assert runtime.portfolio.lean_declarations is not None
    assert (
        runtime.portfolio.lean_checkers[LeanEnvironment.MATHLIB].checker_timeout_seconds
        == 225
    )

    searched = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "MATHLIB",
                "name_contains": "irrational_sqrt_two",
                "result_limit": 1,
            },
        )
    )
    inspected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.inspect",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "MATHLIB",
                "declaration_name": "irrational_sqrt_two",
            },
        )
    )

    assert searched.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert searched.assurance.verification_record_uri is None
    assert searched.output["declarations"][0]["name"] == "irrational_sqrt_two"
    assert inspected.output["declaration"]["type"] == "Irrational √2"
    assert (
        inspected.output["environment_digest"] == searched.output["environment_digest"]
    )

    verified = runtime.portfolio.lean.verify(
        environment=LeanEnvironment.MATHLIB,
        statement="Irrational (Real.sqrt 2)",
        proof="exact irrational_sqrt_two",
    )

    assert verified.result.conclusion is Conclusion.TRUE, verified.result.input.errors
    assert verified.result.assurance.verification is Verification.VERIFIED
    certificate = runtime.core.store.get(verified.certificate_uri)
    assert certificate.payload["payload"]["environment"] == "MATHLIB"
    assert certificate.payload["payload"]["allowed_axioms"] == [
        "Classical.choice",
        "Quot.sound",
        "propext",
    ]
    assert certificate.payload["payload"]["mathlib_commit"] == (
        "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
    )


def test_core_lean_induction_proof_creates_bound_verification_record(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None

    inspected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.inspect",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "declaration_name": "Nat.add",
            },
        )
    )
    outside_profile = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "name_contains": "Lean.Meta.ppExpr",
            },
        )
    )

    assert inspected.output["declaration"]["type"] == "Nat → Nat → Nat"
    assert outside_profile.output["declarations"] == []
    assert outside_profile.output["stop_reason"] == "EXHAUSTED"

    verified = runtime.portfolio.lean.verify(
        statement="∀ n : Nat, n + 0 = n",
        proof=(
            "intro n\n"
            "induction n with\n"
            "| zero => rfl\n"
            "| succ n ih => exact congrArg Nat.succ ih"
        ),
    )

    assert verified.result.conclusion is Conclusion.TRUE
    assert verified.result.assurance.verification is Verification.VERIFIED
    assert verified.result.verification_record_uri is not None
    record = runtime.core.store.get(verified.result.verification_record_uri)
    certificate = runtime.core.store.get(verified.certificate_uri)
    assert record.payload["evidence_uri"] == verified.certificate_uri
    assert record.payload["bindings"] == certificate.payload["bindings"]
    assert set(certificate.manifest.parents) == {
        verified.claim_uri,
        verified.candidate_uri,
    }


@pytest.mark.parametrize(
    ("statement", "proof"),
    (
        ("let n : Nat := 2; n + n = 4", "rfl"),
        ("(fun n : Nat => n + n) 2 = 4", "rfl"),
        ("True", "by trivial"),
    ),
)
def test_core_lean_accepts_single_expression_witness_forms(
    tmp_path: Path,
    statement: str,
    proof: str,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None

    verified = runtime.portfolio.lean.verify(statement=statement, proof=proof)

    assert verified.result.conclusion is Conclusion.TRUE
    assert verified.result.assurance.verification is Verification.VERIFIED


def test_core_lean_check_runs_through_capability_mcp_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path),
            raise_exceptions=True,
        ) as client:
            described = await client.call_tool(
                "math.find",
                {"capability_id": "lean.check", "view": "CONTRACT"},
            )
            assert isinstance(described.structured_content, dict)
            descriptor = described.structured_content
            assert descriptor["invocations"][0]["name"] == "finite-witness-let"
            assert descriptor["cache"]["mathlib_warmup"]["status"] == "NOT_STARTED"

            response = await client.call_tool(
                "math.run",
                {
                    "capability_id": "lean.check",
                    "mode": "VERIFY",
                    "payload": {
                        "statement": "∀ n : Nat, n + 0 = n",
                        "proof": (
                            "intro n\n"
                            "induction n with\n"
                            "| zero => rfl\n"
                            "| succ n ih => exact congrArg Nat.succ ih"
                        ),
                    },
                },
            )
            assert response.is_error is False
            assert isinstance(response.structured_content, dict)
            payload = response.structured_content
            assert payload["output"]["conclusion"] == "TRUE"
            assert payload["assurance"]["level"] == "VERIFIED"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "proof",
    [
        "sorry",
        "native_decide",
        "run_tac exact q(true)",
        "exact Nat.succ.inj",
    ],
)
def test_core_lean_rejects_untrusted_or_invalid_proofs(
    tmp_path: Path,
    proof: str,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None

    rejected = runtime.portfolio.lean.verify(
        statement="∀ n : Nat, n + 0 = n",
        proof=proof,
    )

    assert rejected.result.input.status is InputStatus.REJECTED
    assert rejected.result.conclusion is Conclusion.UNKNOWN
    assert rejected.result.assurance.verification is Verification.UNVERIFIED
    assert rejected.result.verification_record_uri is None


def test_lean_reuses_only_an_exact_active_checker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None
    calls = 0

    def accept(**_: object) -> CheckerDecision:
        nonlocal calls
        calls += 1
        return CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.SYMBOLIC,
            method=Method.CHECKED_CERTIFICATE,
            coverage=Coverage.NOT_APPLICABLE,
            detail="accepted by test checker",
        )

    def unexpected_selector(**_: object) -> object:
        raise AssertionError("Lean must use its explicitly installed checker")

    monkeypatch.setattr(runtime.services.verification, "_run_checker", accept)
    monkeypatch.setattr(runtime.core.checkers, "select_compatible", unexpected_selector)
    first = runtime.portfolio.lean.verify(statement="1 + 1 = 2", proof="rfl")
    repeated = runtime.portfolio.lean.verify(statement="1 + 1 = 2", proof="rfl")
    changed = runtime.portfolio.lean.verify(statement="2 + 2 = 4", proof="rfl")

    assert calls == 2
    assert first.cache_hit is False
    assert repeated.cache_hit is True
    assert repeated.result.verification_record_uri == (
        first.result.verification_record_uri
    )
    assert changed.cache_hit is False


def test_lean_cache_does_not_reuse_a_revoked_checker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None
    monkeypatch.setattr(
        runtime.services.verification,
        "_run_checker",
        lambda **_: CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.SYMBOLIC,
            method=Method.CHECKED_CERTIFICATE,
            coverage=Coverage.NOT_APPLICABLE,
            detail="accepted by test checker",
        ),
    )
    first = runtime.portfolio.lean.verify(statement="1 + 1 = 2", proof="rfl")
    checker_id = runtime.portfolio.lean_checkers[LeanEnvironment.CORE].checker_id
    runtime.core.checkers.revoke(checker_id, reason="cache trust-boundary test")

    repeated = runtime.portfolio.lean.verify(statement="1 + 1 = 2", proof="rfl")

    assert first.result.assurance.verification is Verification.VERIFIED
    assert repeated.cache_hit is False
    assert repeated.result.assurance.verification is Verification.UNVERIFIED


def test_mathlib_warmup_starts_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.portfolio.lean is not None
    warmed = threading.Event()
    monkeypatch.setattr(runtime.portfolio.lean, "_warm_mathlib", warmed.set)

    assert runtime.portfolio.lean.start_mathlib_warmup() is True
    assert warmed.wait(timeout=2)
    assert runtime.portfolio.lean.start_mathlib_warmup() is False
