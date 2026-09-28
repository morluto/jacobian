"""Publication check for the filtered chain-complex direct sum example.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

_OPERATION_ID = "homological.filtered_chain_complex.direct_sum.compute"


def test_direct_sum_example_executes_through_catalog() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["filtered_complex"]["complex"]["basis_sizes"] == [2]
    assert result.output["left"]["complex"]["basis_sizes"] == [1]
    assert result.output["right"]["complex"]["basis_sizes"] == [1]
