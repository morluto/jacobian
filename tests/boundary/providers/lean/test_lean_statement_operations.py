"""Integration tests for Lean statement proposal and comparison."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.provider_lean import (
    PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    pinned_lean_core_runtime_available,
)

import jacobian.lean_frontend.statement as lean_statements
from jacobian.artifacts import ArtifactService
from jacobian.contracts.lean_statement import LeanElaborationOption
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.lean_frontend.statement import (
    LeanStatementCompareAdapter,
    LeanStatementProposalAdapter,
    install_lean_statement_operations,
)
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import project_operation_result
from jacobian.process_policy import ProcessRequest, ProcessResult, ProcessTermination
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

LEAN_AVAILABLE = pinned_lean_core_runtime_available()


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _build_adapters(
    tmp_path: Path,
) -> tuple[
    LeanStatementProposalAdapter,
    LeanStatementCompareAdapter,
]:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    adapters, _installation = install_lean_statement_operations(
        store, schemas, artifacts
    )
    return adapters


# ---------------------------------------------------------------------------
# lean.statement.propose
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not LEAN_AVAILABLE,
    reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
)
def test_propose_elaborates_valid_statement(tmp_path: Path) -> None:
    propose, _ = _build_adapters(tmp_path)

    result = project_operation_result(
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "environment": "CORE",
                        "informal_claim": "one plus one equals two",
                        "proposed_statement": "1 + 1 = 2",
                    },
                )
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["elaborates"] is True
    assert result.output["sorry_count"] == 1
    assert result.output["proposal_uri"] in result.artifact_uris


@pytest.mark.skipif(
    not LEAN_AVAILABLE,
    reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
)
def test_propose_reports_elaboration_failure(tmp_path: Path) -> None:
    propose, _ = _build_adapters(tmp_path)

    result = project_operation_result(
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "environment": "CORE",
                        "informal_claim": "bogus claim",
                        "proposed_statement": "1 + + 1 = 2",
                    },
                )
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["elaborates"] is False
    assert result.output["sorry_count"] == 0
    assert len(result.output["messages"]) > 0


def test_propose_rejects_forbidden_statement(tmp_path: Path) -> None:
    propose, _ = _build_adapters(tmp_path)

    with pytest.raises(OperationInvocationError) as exc_info:
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "environment": "CORE",
                        "informal_claim": "bogus",
                        "proposed_statement": "sorry",
                    },
                )
            )
        )

    assert exc_info.value.diagnostic.code == "INVALID_LEAN_STATEMENT_PROPOSAL"


def test_propose_rejects_mathlib_environment(tmp_path: Path) -> None:
    propose, _ = _build_adapters(tmp_path)

    with pytest.raises(OperationInvocationError) as exc_info:
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "environment": "MATHLIB",
                        "informal_claim": "claim",
                        "proposed_statement": "1 + 1 = 2",
                    },
                )
            )
        )

    assert exc_info.value.diagnostic.code == "INVALID_LEAN_STATEMENT_PROPOSAL"


def test_propose_returns_diagnostic_when_lean_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jacobian_checkers import lean4

    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (_ for _ in ()).throw(
            RuntimeError("pinned Lean unavailable")
        ),
    )
    propose, _ = _build_adapters(tmp_path)

    with pytest.raises(OperationInvocationError) as exc_info:
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "environment": "CORE",
                        "informal_claim": "one plus one equals two",
                        "proposed_statement": "1 + 1 = 2",
                    },
                )
            )
        )

    assert exc_info.value.diagnostic.code == "LEAN_BACKEND_UNAVAILABLE"


def test_propose_directly_elaborates_environment_bound_proposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lean_statements,
        "_elaborate_proposition",
        lambda _statement, **_kwargs: lean_statements._ElaborationResult(
            elaborates=True,
            sorry_count=0,
            messages=(
                "fixture.lean:4:0: info: "
                "@Eq.{1} Nat (@OfNat.ofNat Nat 1 instOfNatNat) "
                "(@OfNat.ofNat Nat 1 instOfNatNat) : Prop",
            ),
            errors=(),
            elaborated_expression=(
                "@Eq.{1} Nat (@OfNat.ofNat Nat 1 instOfNatNat) "
                "(@OfNat.ofNat Nat 1 instOfNatNat)"
            ),
            used_imports=("Init.Prelude",),
            used_declarations=("Eq", "Nat", "OfNat.ofNat", "instOfNatNat"),
            options=(
                LeanElaborationOption(name="pp.all", value="true"),
                LeanElaborationOption(name="pp.explicit", value="true"),
                LeanElaborationOption(name="pp.universes", value="true"),
            ),
        ),
    )
    monkeypatch.setattr(
        lean_statements,
        "_lean_version_info",
        lambda *_args: ("4.31.0", "lean-commit"),
    )
    propose, _ = _build_adapters(tmp_path)

    result = project_operation_result(
        propose.invoke(
            propose.prepare(
                OperationRequest(
                    operation_id="lean.statement.propose",
                    input={
                        "operation": "ELABORATE_PROPOSITION",
                        "environment": "CORE",
                        "proposed_statement": "1 = 1",
                    },
                )
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["operation"] == "ELABORATE_PROPOSITION"
    assert result.output["informal_claim"] is None
    assert result.output["elaborates"] is True
    assert result.output["sorry_count"] == 0
    assert result.output["elaborated_expression"].startswith("@Eq")
    assert result.output["used_imports"] == ["Init.Prelude"]
    assert "Eq" in result.output["used_declarations"]
    assert result.output["options"][0] == {"name": "pp.all", "value": "true"}
    assert result.output["semantic_scope"] == "ELABORATION_ONLY"
    assert result.output["environment_digest"].startswith("sha256:")

    artifact = propose.resources.store.get(result.output["proposal_uri"])
    assert artifact.payload["environment_digest"] == result.output["environment_digest"]
    assert (
        artifact.payload["elaborated_expression"]
        == result.output["elaborated_expression"]
    )


def test_direct_elaboration_parser_preserves_multiline_core_expression() -> None:
    output = (
        "fixture.lean:4:0: info: @Eq.{1} Nat\n"
        "  (@OfNat.ofNat Nat 1 instOfNatNat)\n"
        "  (@OfNat.ofNat Nat 1 instOfNatNat) : Prop\n"
    )

    assert lean_statements._parse_elaborated_expression(output) == (
        "@Eq.{1} Nat (@OfNat.ofNat Nat 1 instOfNatNat) "
        "(@OfNat.ofNat Nat 1 instOfNatNat)"
    )


def test_direct_elaboration_parser_accepts_plain_lean_check_output() -> None:
    assert lean_statements._parse_elaborated_expression("True : Prop\n") == "True"


@pytest.mark.skipif(
    not LEAN_AVAILABLE,
    reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
)
def test_direct_elaboration_core_runtime_matrix() -> None:
    successful_cases = (
        ("True", "True"),
        ("False", "False"),
        ("1 = 1", "@Eq"),
        ("∀ n : Nat, n = n", "∀"),
    )
    for statement, expected_fragment in successful_cases:
        result = lean_statements._elaborate_proposition(statement)
        assert result.elaborates is True
        assert result.errors == ()
        assert result.elaborated_expression is not None
        assert expected_fragment in result.elaborated_expression

    failed_cases = (
        ("NotARealTypeXYZ", "unknown identifier"),
        ("∀ n : Nat,", "unexpected token"),
    )
    for statement, expected_diagnostic in failed_cases:
        result = lean_statements._elaborate_proposition(statement)
        assert result.elaborates is False
        assert result.elaborated_expression is None
        assert any(expected_diagnostic in error.lower() for error in result.errors)


def test_direct_elaboration_parser_preserves_coded_lean_error() -> None:
    output = (
        "fixture.lean:4:8: error(lean.unknownIdentifier): "
        "Unknown identifier `NotARealTypeXYZ`\n"
    )

    assert lean_statements._parse_lean_messages(output) == [output.strip()]
    assert lean_statements._lean_message_severity(output) == "ERROR"


# ---------------------------------------------------------------------------
# lean.statement.compare
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not LEAN_AVAILABLE,
    reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
)
def test_compare_identical_statements(tmp_path: Path) -> None:
    _, compare = _build_adapters(tmp_path)

    result = project_operation_result(
        compare.invoke(
            compare.prepare(
                OperationRequest(
                    operation_id="lean.statement.compare",
                    input={
                        "environment": "CORE",
                        "statement_a": "1 + 1 = 2",
                        "statement_b": "1 + 1 = 2",
                        "axiom_set_a": [],
                        "axiom_set_b": [],
                    },
                )
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["statements_identical"] is True
    assert result.output["axiom_sets_identical"] is True
    assert result.output["elaboration_checked"] is True
    assert result.output["both_elaborate"] is True
    assert result.output["comparison_uri"] in result.artifact_uris


@pytest.mark.skipif(
    not LEAN_AVAILABLE,
    reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
)
def test_compare_different_statements(tmp_path: Path) -> None:
    _, compare = _build_adapters(tmp_path)

    result = project_operation_result(
        compare.invoke(
            compare.prepare(
                OperationRequest(
                    operation_id="lean.statement.compare",
                    input={
                        "environment": "CORE",
                        "statement_a": "1 + 1 = 2",
                        "statement_b": "1 + 1 = 3",
                        "axiom_set_a": ["Classical.choice"],
                        "axiom_set_b": [],
                    },
                )
            )
        )
    )

    assert result.output["statements_identical"] is False
    assert result.output["axiom_sets_identical"] is False


def test_compare_works_without_lean_for_syntactic_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jacobian_checkers import lean4

    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (_ for _ in ()).throw(
            RuntimeError("pinned Lean unavailable")
        ),
    )
    _, compare = _build_adapters(tmp_path)

    result = project_operation_result(
        compare.invoke(
            compare.prepare(
                OperationRequest(
                    operation_id="lean.statement.compare",
                    input={
                        "environment": "CORE",
                        "statement_a": "1 + 1 = 2",
                        "statement_b": "1 + 1 = 2",
                    },
                )
            )
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["statements_identical"] is True
    assert result.output["elaboration_checked"] is False
    assert result.output["both_elaborate"] is False


def test_execution_uses_the_exact_pinned_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian_checkers import lean4

    executable = tmp_path / "pinned-lean"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (executable, None),
    )
    calls: list[list[str]] = []

    def fake_execute(request: ProcessRequest, **_kwargs: object) -> ProcessResult:
        calls.append([request.executable, *request.arguments])
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=b"True : Prop\n",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr(lean_statements, "execute_process", fake_execute)
    result = lean_statements._elaborate_proposition("True")
    version, commit = lean_statements._lean_version_info()

    assert result.elaborates is True
    assert version == "unknown"
    assert commit == "unknown"
    assert calls
    assert all(call[0] == str(executable) for call in calls)


def test_stale_pinned_executable_becomes_backend_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_executable = tmp_path / "removed-lean"

    def fake_execute(_request: ProcessRequest, **_kwargs: object) -> ProcessResult:
        raise FileNotFoundError(stale_executable)

    monkeypatch.setattr(lean_statements, "execute_process", fake_execute)

    with pytest.raises(lean_statements._LeanUnavailableError, match="could not run"):
        lean_statements._execute_lean_source(
            "#check True",
            executable=str(stale_executable),
            timeout_seconds=1,
        )


def test_replaced_pinned_executable_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.providers.lean_runtime import lean_frontend_provider_runtime
    from jacobian_checkers import lean4

    executable = tmp_path / "pinned-lean"
    executable.write_bytes(b"original")
    monkeypatch.setattr(
        lean4,
        "inspect_runtime",
        lambda *, require_mathlib: (executable, None),
    )
    runtime = lean_frontend_provider_runtime()
    executable.write_bytes(b"replacement")

    def unexpected_execute(*_args: object, **_kwargs: object) -> ProcessResult:
        pytest.fail("replaced Lean executable was launched")

    monkeypatch.setattr(lean_statements, "execute_process", unexpected_execute)

    with pytest.raises(
        lean_statements._LeanUnavailableError,
        match="identity changed",
    ):
        lean_statements._execute_lean_source(
            "#check True",
            executable=str(executable),
            provider_runtime=runtime,
            timeout_seconds=1,
        )


def test_compare_normalizes_whitespace(tmp_path: Path) -> None:
    _, compare = _build_adapters(tmp_path)

    result = project_operation_result(
        compare.invoke(
            compare.prepare(
                OperationRequest(
                    operation_id="lean.statement.compare",
                    input={
                        "environment": "CORE",
                        "statement_a": "1 + 1  =  2",
                        "statement_b": "1 + 1 = 2",
                    },
                )
            )
        )
    )

    assert result.output["statements_identical"] is True


def test_compare_rejects_forbidden_statement(tmp_path: Path) -> None:
    _, compare = _build_adapters(tmp_path)

    with pytest.raises(OperationInvocationError) as exc_info:
        compare.invoke(
            compare.prepare(
                OperationRequest(
                    operation_id="lean.statement.compare",
                    input={
                        "environment": "CORE",
                        "statement_a": "sorry",
                        "statement_b": "True",
                    },
                )
            )
        )

    assert exc_info.value.diagnostic.code == "INVALID_LEAN_STATEMENT_COMPARISON"


# ---------------------------------------------------------------------------
# Descriptor checks.
# ---------------------------------------------------------------------------


def test_descriptors_have_correct_ids(tmp_path: Path) -> None:
    propose, compare = _build_adapters(tmp_path)

    assert propose.descriptor.operation_id == "lean.statement.propose"
    assert propose.descriptor.version == "2"
    assert compare.descriptor.operation_id == "lean.statement.compare"
