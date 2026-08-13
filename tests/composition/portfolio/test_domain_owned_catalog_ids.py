from __future__ import annotations

from jacobian.runtime.model import JacobianRuntime


def test_runtime_catalog_uses_only_domain_owned_operation_ids(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    catalog_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
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
