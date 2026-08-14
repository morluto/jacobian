from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.catalog_build_runtime import create_catalog_build_runtime
from tests.support.state import copy_template

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_measurements import _cold_install_spec


def test_unhealthy_optional_lean_runtime_is_absent_from_catalog(
    tmp_path: Path,
    authorized_portfolio_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = copy_template(authorized_portfolio_template, tmp_path / "state")
    unavailable = ProviderObservation(
        provider="jacobian.lean4",
        availability=ProviderAvailability.UNAVAILABLE,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T3,
        license_id="Apache-2.0",
        diagnostic="The pinned Lean runtime is unavailable.",
    )
    monkeypatch.setattr(
        "jacobian.catalog_checkers.lean_provider_runtime",
        lambda **_kwargs: unavailable,
    )

    runtime = create_catalog_build_runtime(
        state, checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING
    )
    try:
        assert runtime.catalog_build_resources.lean is None
        operation_ids = {
            item.operation_id for item in runtime.core.operations.snapshot().operations
        }
        assert {
            "lean.check",
            "lean.declaration.dependencies",
            "lean.declaration.inspect",
            "lean.declaration.search",
            "lean.proof.axioms.inspect",
            "lean.proof_edit.validate",
        }.isdisjoint(operation_ids)
    finally:
        runtime.close()


def test_unhealthy_lean_frontend_is_absent_from_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = ProviderObservation(
        provider="jacobian.lean4",
        availability=ProviderAvailability.UNAVAILABLE,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T3,
        license_id="Apache-2.0",
        diagnostic=("TOOLCHAIN_RESOLUTION: the pinned Lean executable is unavailable"),
    )
    monkeypatch.setattr(
        "jacobian.catalog_resources.lean_frontend_provider_runtime",
        lambda: unavailable,
    )

    runtime = create_catalog_build_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.NONE
    )
    try:
        operation_ids = {
            item.operation_id for item in runtime.core.operations.snapshot().operations
        }
        assert {"lean.statement.propose", "lean.statement.compare"}.isdisjoint(
            operation_ids
        )
    finally:
        runtime.close()


def test_source_runtime_has_no_implicit_working_directory_install() -> None:
    runtime = ProviderObservation(
        provider="tests.fixture",
        availability=ProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "a" * 64,
        digest_kind=ProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
    )

    assert _cold_install_spec(runtime) is None
