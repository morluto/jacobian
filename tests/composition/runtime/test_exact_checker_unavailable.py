from __future__ import annotations

from pathlib import Path

import pytest

import jacobian.exact_domain_checkers as exact_domain_checkers
from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.providers.flint_runtime import exact_domain_checker_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime


def test_unavailable_flint_replay_preserves_runtime_and_reports_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = exact_domain_checker_provider_runtime().model_copy(
        update={
            "availability": CapabilityProviderAvailability.UNAVAILABLE,
            "version": None,
            "digest": None,
            "digest_kind": None,
            "diagnostic": "The optional python-flint backend is unavailable.",
        }
    )
    monkeypatch.setattr(
        exact_domain_checkers,
        "exact_domain_checker_provider_runtime",
        lambda **_: unavailable,
    )

    runtime = create_runtime(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    try:
        capability_ids = {
            descriptor.capability_id
            for descriptor in runtime.core.capabilities.catalog().capabilities
        }
        assert "matrix.normal_form.rref.compute" in capability_ids
        assert "matrix.normal_form.rref.verify" not in capability_ids
        assert "graph.hamiltonian_path.verify" in capability_ids

    finally:
        runtime.close()
