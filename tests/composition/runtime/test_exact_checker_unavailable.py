from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.domains.graph_optimization import build_graph_optimization_bundle
from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.providers.flint_runtime import exact_domain_checker_provider_runtime


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
        "jacobian.providers.flint_runtime.exact_domain_checker_provider_runtime",
        lambda **_: unavailable,
    )

    with open_exact_domain_services(
        tmp_path / "state",
        build_matrix_bundle(),
        build_graph_optimization_bundle(),
    ) as services:
        capability_ids = {
            descriptor.capability_id
            for descriptor in services.core.capabilities.catalog().capabilities
        }
        assert "matrix.normal_form.rref.compute" in capability_ids
        assert "matrix.normal_form.rref.verify" not in capability_ids
        assert "graph.hamiltonian_path.verify" in capability_ids
