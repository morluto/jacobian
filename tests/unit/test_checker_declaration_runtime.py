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


def test_exact_replay_declaration_owns_unassigned_provider_runtime() -> None:
    runtime = _runtime()
    declaration = ExactReplayCheckerDeclaration(
        "test.compute.value",
        CanonicalRational,
        "check_rational_solution",
        "test.value.replay",
        entrypoint_module="jacobian_checkers.linear",
        provider_runtime=runtime,
    )

    assert declaration.provider_runtime == runtime
    assert declaration.provider_runtime.checker_ids == ()


def test_exact_replay_declaration_rejects_preauthorized_runtime() -> None:
    runtime = _runtime().model_copy(update={"checker_ids": ("checker:test",)})

    with pytest.raises(ValueError, match="must not pre-authorize checker IDs"):
        ExactReplayCheckerDeclaration(
            "test.compute.value",
            CanonicalRational,
            "check_rational_solution",
            "test.value.replay",
            entrypoint_module="jacobian_checkers.linear",
            provider_runtime=runtime,
        )
