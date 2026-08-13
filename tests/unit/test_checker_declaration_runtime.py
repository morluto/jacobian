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


def _declaration(**kwargs):
    return ExactReplayCheckerDeclaration(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        **kwargs,
    )


def test_exact_replay_declaration_defers_and_caches_provider_runtime() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _runtime()

    declaration = _declaration(provider_runtime_factory=factory)

    assert calls == 0
    runtime = declaration.provider_runtime
    assert calls == 1
    assert runtime is not None
    assert runtime.checker_ids == ()
    assert declaration.provider_runtime is runtime
    assert calls == 1


def test_generated_repr_and_equality_do_not_realize_provider_runtime() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _runtime()

    declaration = _declaration(provider_runtime_factory=factory)
    equivalent = _declaration(provider_runtime_factory=factory)

    assert "provider_runtime" not in repr(declaration)
    assert declaration == equivalent
    assert calls == 0


def test_exact_replay_declaration_keeps_direct_runtime_compatibility() -> None:
    runtime = _runtime()
    declaration = _declaration(provider_runtime=runtime)

    assert declaration.provider_runtime is runtime


def test_exact_replay_declaration_rejects_two_runtime_owners() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="either provider_runtime"):
        _declaration(
            provider_runtime=runtime,
            provider_runtime_factory=lambda: runtime,
        )


def test_exact_replay_declaration_rejects_preauthorized_direct_runtime() -> None:
    runtime = _runtime().model_copy(update={"checker_ids": ("checker:test",)})

    with pytest.raises(ValueError, match="must not pre-authorize checker IDs"):
        _declaration(provider_runtime=runtime)


def test_exact_replay_declaration_rejects_preauthorized_realized_runtime() -> None:
    runtime = _runtime().model_copy(update={"checker_ids": ("checker:test",)})
    declaration = _declaration(provider_runtime_factory=lambda: runtime)

    with pytest.raises(ValueError, match="must not pre-authorize checker IDs"):
        _ = declaration.provider_runtime


def test_exact_replay_declaration_rejects_wrong_runtime_factory_result() -> None:
    declaration = _declaration(
        provider_runtime_factory=lambda: object(),  # type: ignore[return-value]
    )

    with pytest.raises(TypeError, match="must return CapabilityProviderRuntime"):
        _ = declaration.provider_runtime
