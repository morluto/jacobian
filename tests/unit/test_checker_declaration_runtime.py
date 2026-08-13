from __future__ import annotations

import pytest

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityInstallTier
from jacobian.contracts.exact import CanonicalRational
from jacobian.provider_runtime import source_provider_runtime


def _runtime():
    return source_provider_runtime(
        "jacobian.test-declaration-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_solution",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        features=("clean-process-checker",),
    )


def test_exact_replay_declaration_defers_and_caches_provider_runtime() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _runtime()

    declaration = ExactReplayCheckerDeclaration(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        provider_runtime_factory=factory,
    )

    assert calls == 0
    runtime = declaration.provider_runtime
    assert calls == 1
    assert runtime is not None
    assert runtime.checker_ids == ()
    assert declaration.provider_runtime is runtime
    assert calls == 1


def test_exact_replay_declaration_rejects_preauthorized_realized_runtime() -> None:
    runtime = _runtime().model_copy(update={"checker_ids": ("checker:test",)})
    declaration = ExactReplayCheckerDeclaration(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        provider_runtime_factory=lambda: runtime,
    )

    with pytest.raises(ValueError, match="must not pre-authorize checker IDs"):
        _ = declaration.provider_runtime


def test_exact_replay_declaration_rejects_wrong_runtime_factory_result() -> None:
    declaration = ExactReplayCheckerDeclaration(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        provider_runtime_factory=lambda: object(),  # type: ignore[return-value]
    )

    with pytest.raises(TypeError, match="must return CapabilityProviderRuntime"):
        _ = declaration.provider_runtime
