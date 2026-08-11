from __future__ import annotations

from pathlib import Path

import pytest

import jacobian.exact_domain_checkers as exact_domain_checkers
from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.providers.flint_runtime import exact_domain_checker_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime


def _bundles(
    runtime: JacobianRuntime, *domain_ids: str
) -> dict[str, tuple[object, object]]:
    portfolio = build_builtin_portfolio()
    return {
        domain_id: (
            portfolio.bundle_for(domain_id),
            runtime.portfolio.domain_bundles[domain_id],
        )
        for domain_id in domain_ids
    }


def _install_verification(
    fresh_complete_runtime: JacobianRuntime, *, authorize: bool
) -> tuple[object, ...]:
    adapters, _ = install_exact_domain_verification(
        fresh_complete_runtime.core.store,
        fresh_complete_runtime.core.schemas,
        fresh_complete_runtime.core.artifacts,
        fresh_complete_runtime.services.verification,
        fresh_complete_runtime.core.checkers,
        bundles=_bundles(fresh_complete_runtime, "polynomial", "matrix", "probability"),
        authorize=authorize,
    )
    for adapter in adapters:
        fresh_complete_runtime.core.capabilities.register(adapter)
    return adapters


def test_probability_verification_installs_without_polynomial_or_matrix_bundles(
    fresh_complete_runtime,
) -> None:
    adapters, installation = install_exact_domain_verification(
        fresh_complete_runtime.core.store,
        fresh_complete_runtime.core.schemas,
        fresh_complete_runtime.core.artifacts,
        fresh_complete_runtime.services.verification,
        fresh_complete_runtime.core.checkers,
        bundles=_bundles(fresh_complete_runtime, "probability"),
        authorize=True,
    )

    assert [adapter.descriptor.capability_id for adapter in adapters] == [
        "probability.finite_distribution.raw_moment.verify",
        "probability.finite_distribution.event_probability.verify",
        "probability.finite_distribution.condition.verify",
        "probability.finite_distribution.pushforward.verify",
        "probability.finite_distribution.convolution.verify",
        "probability.gaussian_polynomial.moment.verify",
        "probability.graph_reliability.connection_probability.verify",
    ]
    assert any(installation.checker_ids.values())


def test_operator_can_leave_exact_result_verification_unavailable(
    fresh_complete_runtime,
) -> None:

    adapters = _install_verification(fresh_complete_runtime, authorize=False)

    assert adapters == ()
    assert {"polynomial.gcd.verify", "matrix.multiply.verify"}.isdisjoint(
        {
            descriptor.capability_id
            for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
        }
    )


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

        installation = runtime.portfolio.exact_domain_checkers
        assert installation is not None
        assert any(
            diagnostic.details.get("capability_id") == "matrix.normal_form.rref.compute"
            for diagnostic in installation.diagnostics
        )
    finally:
        runtime.close()
