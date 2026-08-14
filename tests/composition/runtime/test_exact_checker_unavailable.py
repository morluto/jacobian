from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.contracts.operations import ProviderAvailability
from jacobian.domains.graph_optimization import graph_optimization_operations
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.providers.flint_runtime import exact_domain_checker_provider_runtime


def test_unavailable_flint_replay_preserves_runtime_and_reports_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = exact_domain_checker_provider_runtime().model_copy(
        update={
            "availability": ProviderAvailability.UNAVAILABLE,
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
        matrix_operations(),
        graph_optimization_operations(),
    ) as services:
        operation_ids = {
            descriptor.operation_id
            for descriptor in services.core.operations.snapshot().operations
        }
        assert "matrix.normal_form.rref.compute" in operation_ids
        assert "matrix.normal_form.rref.verify" not in operation_ids
        assert "graph.hamiltonian_path.verify" in operation_ids
