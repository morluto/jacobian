"""Composition wiring: builtin runtime catalogs domain-owned operation IDs."""

from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.runtime import create_runtime

COMPOSITION_ADMISSION = "WIRING"


@pytest.mark.composition_admission("WIRING")
def test_runtime_catalog_uses_only_domain_owned_operation_ids(tmp_path: Path) -> None:
    catalog_ids = {
        descriptor.capability_id
        for descriptor in create_runtime(tmp_path)
        .core.capabilities.catalog()
        .capabilities
    }

    assert {
        "number_theory.compute.jacobi_symbol",
        "modular.compute.discrete_logarithm",
        "combinatorics.enumerate.integer_partitions",
    } <= catalog_ids
    assert {
        "number_theory.jacobi_symbol.compute",
        "number_theory.discrete_log.bounded",
        "combinatorics.integer_partition.enumerate",
    }.isdisjoint(catalog_ids)
