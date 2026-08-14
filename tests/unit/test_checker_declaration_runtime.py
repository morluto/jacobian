from __future__ import annotations

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.operations import ProviderInstallTier
from jacobian.provider_runtime import source_provider_runtime


def _runtime():
    return source_provider_runtime(
        "jacobian.test-declaration-checker",
        version="1",
        entrypoint="jacobian_checkers.linear:check_rational_solution",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("clean-process-checker",),
    )


def _declaration(**kwargs):
    return AuthorizedChecker(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        **kwargs,
    )


def test_authorized_checker_keeps_observation_loading_explicit() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _runtime()

    declaration = _declaration(observation_loader=factory)

    assert calls == 0
    runtime = declaration.observation_loader()
    assert calls == 1
    assert runtime.checker_ids == ()
    assert declaration.observation_loader() is not runtime
    assert calls == 2


def test_generated_repr_and_equality_do_not_realize_provider_runtime() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _runtime()

    declaration = _declaration(observation_loader=factory)
    equivalent = _declaration(observation_loader=factory)

    assert "observation_loader" in repr(declaration)
    assert declaration == equivalent
    assert calls == 0
