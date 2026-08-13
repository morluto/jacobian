from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateRequest
from jacobian.lean_frontend.exploration import _resolve_typed_goal_helper, _Resources
from jacobian.lean_frontend.premise_retrieval import LeanPremiseRetrievalAdapter
from jacobian.lean_frontend.proof_state import LeanProofStateAdapter
from jacobian.lean_frontend.repl import _single_proof_state
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandResponse,
    LeanReplProofStepResponse,
)


def test_typed_goal_helper_derives_default_elan_home_without_forwarding_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "operator-home"
    runtime = tmp_path / "lean-runtime"
    helper = runtime / ".lake" / "build" / "bin" / "jacobian_lean_proof_state"
    helper.parent.mkdir(parents=True)
    helper.touch()
    monkeypatch.delenv("ELAN_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration.require_lean_semantic_runtime_identity",
        lambda _runtime: None,
    )
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration.lean_mathlib_git_config",
        lambda _runtime: {},
    )
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration.shutil.which",
        lambda name: "/usr/bin/elan" if name == "elan" else None,
    )
    resources = cast(
        _Resources,
        SimpleNamespace(
            runtime=runtime,
            provider_runtime=object(),
            installations={
                LeanEnvironment.CORE: SimpleNamespace(lean_version="4.31.0")
            },
        ),
    )

    _elan, _arguments, environment = _resolve_typed_goal_helper(
        resources,
        LeanProofStateRequest(
            environment=LeanEnvironment.CORE,
            statement="True",
            tactic="trivial",
        ),
        tmp_path / "query.json",
    )

    assert environment["ELAN_HOME"] == str(home / ".elan")
    assert "HOME" not in environment


def test_typed_goal_extraction_failure_is_a_structured_non_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
    )
    installation = SimpleNamespace(
        lean_version="4.31.0",
        lean_commit="abc",
        mathlib_commit=None,
    )
    resources = cast(
        _Resources,
        SimpleNamespace(
            installations={LeanEnvironment.CORE: installation},
            provider_runtime=runtime,
            repl=SimpleNamespace(
                execute_clean=lambda **_: (
                    LeanReplCommandResponse.model_validate(
                        {
                            "env": 0,
                            "sorries": [{"goal": "⊢ True", "proofState": 0}],
                        }
                    ),
                    LeanReplProofStepResponse.model_validate(
                        {
                            "proofState": 0,
                            "goals": ["⊢ True"],
                            "proofStatus": "InProgress",
                        }
                    ),
                    LeanReplProofStepResponse.model_validate(
                        {
                            "proofState": 1,
                            "goals": ["⊢ True"],
                            "proofStatus": "InProgress",
                        }
                    ),
                )
            ),
        ),
    )

    def fail_extraction(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("helper protocol failed")

    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration._extract_typed_goals",
        fail_extraction,
    )

    adapter = LeanProofStateAdapter(resources)
    with pytest.raises(CapabilityInvocationError) as error:
        adapter.invoke(
            adapter.prepare(
                CapabilityRequest(
                    capability_id="lean.proof_state.apply_tactic",
                    input={"statement": "True", "tactic": "skip"},
                )
            )
        )

    assert error.value.diagnostic.code == "LEAN_PROOF_STATE_EXTRACTION_FAILED"
    assert error.value.diagnostic.stage == "proof_state_extraction"


def test_single_proof_state_raises_with_lean_errors() -> None:
    response = LeanReplCommandResponse.model_validate(
        {
            "env": 0,
            "sorries": [],
            "messages": [
                {
                    "pos": {"line": 0, "column": 0},
                    "severity": "error",
                    "data": "unknown identifier",
                },
                {
                    "pos": {"line": 0, "column": 0},
                    "severity": "error",
                    "data": "type mismatch",
                },
            ],
        }
    )

    with pytest.raises(RuntimeError, match=r"unknown identifier.*type mismatch"):
        _single_proof_state(response)


def test_premise_retrieval_rejects_by_prefix_before_starting_lean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="4.31.0",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="test",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
    )
    resources = cast(
        _Resources,
        SimpleNamespace(provider_runtime=runtime),
    )

    def unexpected_repl(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid proof prefixes must not start Lean")

    monkeypatch.setattr(
        "jacobian.lean_frontend.premise_retrieval._run_repl",
        unexpected_repl,
    )

    adapter = LeanPremiseRetrievalAdapter(resources)
    example = adapter.descriptor.invocation_examples[0].input
    assert example["proof_prefix"] == ["intro x"]
    assert adapter.descriptor.input_schema["properties"]["proof_prefix"][
        "description"
    ].endswith("Do not include `by`.")

    with pytest.raises(CapabilityInvocationError) as error:
        adapter.invoke(
            adapter.prepare(
                CapabilityRequest(
                    capability_id="lean.retrieve.premises",
                    input={
                        "environment": "MATHLIB",
                        "statement": "∀ x : Real, x ^ 2 ≥ 0",
                        "proof_prefix": ["by", "intro x"],
                    },
                )
            )
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "INVALID_LEAN_RETRIEVAL_REQUEST"
    assert diagnostic.stage == "request_validation"
    assert diagnostic.path == "proof_prefix.0"
    assert "must not include `by`" in diagnostic.message
    assert diagnostic.details["validation_errors"][0]["path"] == "proof_prefix.0"


def test_premise_retrieval_maps_repl_runtime_failure_to_domain_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="4.31.0",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="test",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
    )
    resources = cast(
        _Resources,
        SimpleNamespace(
            provider_runtime=runtime,
            installations={
                LeanEnvironment.MATHLIB: SimpleNamespace(
                    lean_version="4.31.0",
                    lean_commit="lean-commit",
                    mathlib_commit="mathlib-commit",
                )
            },
        ),
    )

    def fail_repl(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Lean did not expose one replayable proof state")

    monkeypatch.setattr(
        "jacobian.lean_frontend.premise_retrieval._run_repl",
        fail_repl,
    )

    adapter = LeanPremiseRetrievalAdapter(resources)
    with pytest.raises(CapabilityInvocationError) as error:
        adapter.invoke(
            adapter.prepare(
                CapabilityRequest(
                    capability_id="lean.retrieve.premises",
                    input={"statement": "True"},
                )
            )
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "LEAN_RETRIEVAL_FAILED"
    assert diagnostic.stage == "premise_retrieval"
    assert diagnostic.details["raw_backend_message"] == (
        "Lean did not expose one replayable proof state"
    )
