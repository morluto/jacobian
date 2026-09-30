"""Distinct multiplicative normal forms retain their public entry points."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.arithmetic._multiplicative_forms import (
    IntegerRequest,
    NonnegativeIntegerRequest,
)


@pytest.mark.parametrize(
    ("value", "kind", "coefficient", "radicand"),
    [
        (0, "ZERO", 0, 1),
        (72, "IRRATIONAL_QUADRATIC", 6, 2),
        (144, "RATIONAL_INTEGER", 12, 1),
    ],
)
def test_quadratic_radical_remains_a_distinct_public_operation(
    value: int, kind: str, coefficient: int, radicand: int
) -> None:
    tool = Catalog.open().operation(
        "quadratic_radical.positive_integer.normalize.compute"
    )
    assert tool is not None
    result = tool.run(NonnegativeIntegerRequest(value=value))
    assert (result.kind, result.coefficient, result.radicand) == (
        kind,
        coefficient,
        radicand,
    )
    assert result.coefficient**2 * result.radicand == value
    assert tool.result_type.model_validate_json(result.model_dump_json()) == result


def test_squarefree_decomposition_keeps_zero_and_nonzero_contracts() -> None:
    catalog = Catalog.open()
    assert catalog.operation("integer.squarefree_part.compute") is None
    tool = catalog.operation("integer.squarefree_decomposition.compute")
    assert tool is not None
    assert "Zero returns a ZERO variant" in tool.description
    zero = tool.run(IntegerRequest(value=0))
    assert zero.kind == "ZERO"
    assert zero.square_factor is None
    assert zero.squarefree_part is None
    result = tool.run(IntegerRequest(value=-72))
    assert (result.square_factor, result.squarefree_part, result.reconstruction) == (
        6,
        -2,
        -72,
    )
