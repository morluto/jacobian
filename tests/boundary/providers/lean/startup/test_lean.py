from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.catalog_build_runtime import create_catalog_build_runtime
from tests.support.provider_lean import (
    PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    skip_unless_pinned_lean_core_runtime,
    skip_unless_pinned_mathlib_runtime,
)
from tests.support.state import copy_template

from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.lean import (
    LeanDeclarationSearchRequest,
    LeanEnvironment,
)
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    Coverage,
    InputStatus,
    Method,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationBackendError,
    LeanSubprocessDeclarationBackend,
)
from jacobian.operator_lifecycle import initialize_state

pytestmark = [
    pytest.mark.skipif(
        skip_unless_pinned_lean_core_runtime(),
        reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    ),
]


def test_core_declaration_catalog_matches_a_fresh_scan_and_detects_tampering(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> None:
    indexed_root = tmp_path / "indexed"
    fresh_root = tmp_path / "fresh"
    copy_template(authorized_portfolio_template, indexed_root)
    copy_template(authorized_portfolio_template, fresh_root)
    indexed_runtime = create_catalog_build_runtime(
        indexed_root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    fresh_runtime = create_catalog_build_runtime(
        fresh_root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    indexed = indexed_runtime.catalog_build_resources.lean_declarations
    fresh = fresh_runtime.catalog_build_resources.lean_declarations
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
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )

    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lean.declaration.dependencies",
            input={
                "environment": "CORE",
                "root_declaration": "Nat.add_comm",
                "max_depth": 1,
                "max_nodes": 40,
            },
        )
    )

    preview = result.output["preview"]
    assert preview["nodes"][0] == {
        "name": "Nat.add_comm",
        "kind": "THEOREM",
        "depth": 0,
    }
    assert len(preview["nodes"]) <= 40
    assert result.output["result_uri"] in result.artifact_uris
    artifact = runtime.core.store.get(result.output["result_uri"])
    assert artifact.payload["nodes"][0]["name"] == "Nat.add_comm"
    assert artifact.payload["query"]["max_depth"] == 1


@pytest.mark.skipif(
    skip_unless_pinned_mathlib_runtime(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
@pytest.mark.timeout(240)
def test_mathlib_discovery_composes_with_bound_sqrt_two_verification(
    tmp_path: Path,
) -> None:
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None
    assert runtime.catalog_build_resources.lean_declarations is not None

    searched = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lean.declaration.search",
            input={
                "environment": "MATHLIB",
                "name_contains": "irrational_sqrt_two",
                "result_limit": 1,
            },
        )
    )
    inspected = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lean.declaration.inspect",
            input={
                "environment": "MATHLIB",
                "declaration_name": "irrational_sqrt_two",
            },
        )
    )

    assert searched.output["result"]["declarations"][0]["name"] == (
        "irrational_sqrt_two"
    )
    assert inspected.output["result"]["declaration"]["type"] == "Irrational √2"
    assert (
        inspected.output["result"]["environment_digest"]
        == searched.output["result"]["environment_digest"]
    )

    verified = runtime.catalog_build_resources.lean.verify(
        environment=LeanEnvironment.MATHLIB,
        statement="Irrational (Real.sqrt 2)",
        proof="exact irrational_sqrt_two",
    )

    assert verified.result.conclusion is Conclusion.TRUE, verified.result.input.errors
    assert verified.result.verification_record_uri is not None
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
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None

    inspected = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lean.declaration.inspect",
            input={
                "environment": "CORE",
                "declaration_name": "Nat.add",
            },
        )
    )
    outside_profile = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lean.declaration.search",
            input={
                "environment": "CORE",
                "name_contains": "Lean.Meta.ppExpr",
            },
        )
    )

    assert inspected.output["result"]["declaration"]["type"] == "Nat → Nat → Nat"
    assert outside_profile.output["result"]["declarations"] == []
    assert outside_profile.output["result"]["stop_reason"] == "EXHAUSTED"

    verified = runtime.catalog_build_resources.lean.verify(
        statement="∀ n : Nat, n + 0 = n",
        proof=(
            "intro n\n"
            "induction n with\n"
            "| zero => rfl\n"
            "| succ n ih => exact congrArg Nat.succ ih"
        ),
    )

    assert verified.result.conclusion is Conclusion.TRUE
    assert verified.result.verification_record_uri is not None
    assert verified.result.verification_record_uri is not None
    record = runtime.core.store.get(verified.result.verification_record_uri)
    certificate = runtime.core.store.get(verified.certificate_uri)
    assert record.payload["evidence_uri"] == verified.certificate_uri
    assert record.payload["bindings"] == certificate.payload["bindings"]
    assert set(certificate.manifest.parents) == {
        verified.claim_uri,
        verified.candidate_uri,
    }


def test_core_lean_checker_binds_the_measured_runtime(tmp_path: Path) -> None:
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None
    installation = runtime.catalog_build_resources.lean.installations[
        LeanEnvironment.CORE
    ]
    assert installation.checker_id is not None
    registration = runtime.core.checkers.require_active(installation.checker_id)

    assert registration.implementation.provider_runtime is not None
    assert registration.implementation.provider_runtime.provider == "jacobian.lean4"
    assert registration.implementation.provider_runtime.checker_ids == ()


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
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None

    verified = runtime.catalog_build_resources.lean.verify(
        statement=statement, proof=proof
    )

    assert verified.result.conclusion is Conclusion.TRUE
    assert verified.result.verification_record_uri is not None


def test_core_lean_check_runs_through_operation_mcp_surface(tmp_path: Path) -> None:
    initialize_state(tmp_path)

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path),
            raise_exceptions=True,
        ) as client:
            described = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "lean.check",
                    }
                },
            )
            assert isinstance(described.structured_content, dict)
            descriptor = described.structured_content
            assert descriptor["operation"]["examples"][0]["name"] == (
                "finite-witness-let"
            )

            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "lean.check",
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
            assert payload["verification_record_uri"] is not None

    asyncio.run(scenario())


@pytest.mark.skipif(
    skip_unless_pinned_mathlib_runtime(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
@pytest.mark.timeout(240)
def test_mathlib_lean_check_uses_its_compiled_checker_binding(tmp_path: Path) -> None:
    initialize_state(tmp_path)

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "lean.check",
                    "payload": {
                        "environment": "MATHLIB",
                        "statement": "Irrational (Real.sqrt 2)",
                        "proof": "exact irrational_sqrt_two",
                    },
                },
            )

            assert response.is_error is False
            assert isinstance(response.structured_content, dict)
            payload = response.structured_content
            assert payload["output"]["conclusion"] == "TRUE"
            assert payload["verification_record_uri"] is not None

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
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None

    rejected = runtime.catalog_build_resources.lean.verify(
        statement="∀ n : Nat, n + 0 = n",
        proof=proof,
    )

    assert rejected.result.input.status is InputStatus.REJECTED
    assert rejected.result.conclusion is Conclusion.UNKNOWN
    assert rejected.result.verification_record_uri is None
    assert rejected.result.verification_record_uri is None
    assert rejected.diagnostics
    diagnostic = rejected.diagnostics[0]
    assert diagnostic.code.startswith("LEAN_")
    assert diagnostic.phase.value == "KERNEL_CHECK"
    assert diagnostic.raw_backend_message


def test_lean_reuses_only_an_exact_active_checker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None
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

    monkeypatch.setattr(runtime.verification._checker_executor, "execute", accept)
    monkeypatch.setattr(runtime.core.checkers, "select_compatible", unexpected_selector)
    first = runtime.catalog_build_resources.lean.verify(
        statement="1 + 1 = 2", proof="rfl"
    )
    repeated = runtime.catalog_build_resources.lean.verify(
        statement="1 + 1 = 2", proof="rfl"
    )
    changed = runtime.catalog_build_resources.lean.verify(
        statement="2 + 2 = 4", proof="rfl"
    )

    assert calls == 2
    assert first.cache_hit is False
    assert repeated.cache_hit is True
    assert repeated.result.verification_record_uri == (
        first.result.verification_record_uri
    )
    assert changed.cache_hit is False


def test_lean_cache_never_reuses_a_rejected_checker_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None
    decisions = iter(
        (
            CheckerDecision(
                accepted=False,
                conclusion=Conclusion.UNKNOWN,
                arithmetic=Arithmetic.SYMBOLIC,
                method=Method.CHECKED_CERTIFICATE,
                coverage=Coverage.NOT_APPLICABLE,
                detail="transient checker setup failure",
            ),
            CheckerDecision(
                accepted=True,
                conclusion=Conclusion.TRUE,
                arithmetic=Arithmetic.SYMBOLIC,
                method=Method.CHECKED_CERTIFICATE,
                coverage=Coverage.NOT_APPLICABLE,
                detail="accepted after checker recovery",
            ),
        )
    )
    calls = 0

    def recover(**_: object) -> CheckerDecision:
        nonlocal calls
        calls += 1
        return next(decisions)

    monkeypatch.setattr(runtime.verification._checker_executor, "execute", recover)

    first = runtime.catalog_build_resources.lean.verify(
        statement="True", proof="by trivial"
    )
    recovered = runtime.catalog_build_resources.lean.verify(
        statement="True", proof="by trivial"
    )
    repeated = runtime.catalog_build_resources.lean.verify(
        statement="True", proof="by trivial"
    )

    assert first.result.input.status is InputStatus.REJECTED
    assert first.cache_hit is False
    assert recovered.result.verification_record_uri is not None
    assert recovered.cache_hit is False
    assert repeated.cache_hit is True
    assert calls == 2


def test_lean_cache_does_not_reuse_a_revoked_checker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )
    assert runtime.catalog_build_resources.lean is not None
    monkeypatch.setattr(
        runtime.verification._checker_executor,
        "execute",
        lambda **_: CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.SYMBOLIC,
            method=Method.CHECKED_CERTIFICATE,
            coverage=Coverage.NOT_APPLICABLE,
            detail="accepted by test checker",
        ),
    )
    first = runtime.catalog_build_resources.lean.verify(
        statement="1 + 1 = 2", proof="rfl"
    )
    record_uri = first.result.verification_record_uri
    assert record_uri is not None
    checker_id = runtime.core.store.get(record_uri).payload["checker_id"]
    runtime.core.checkers.revoke(checker_id, reason="cache trust-boundary test")

    repeated = runtime.catalog_build_resources.lean.verify(
        statement="1 + 1 = 2", proof="rfl"
    )

    assert first.result.verification_record_uri is not None
    assert repeated.cache_hit is False
    assert repeated.result.verification_record_uri is None
