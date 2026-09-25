"""Catalog behavior for nonnegative Petri invariant generators."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_nonunimodular_hilbert_basis_example_executes_through_catalog() -> None:
    catalog = Catalog.open()
    operation_id = "petri_net.nonnegative_invariant_generators.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    assert len(operation.examples) == 1

    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    assert result.output["t_generators"] == [[0, 5, 3], [1, 1, 1], [5, 0, 2]]
    assert result.output["p_generators"] == []
    assert operation.result_type.model_validate(result.output).model_dump(
        mode="json"
    ) == result.output
