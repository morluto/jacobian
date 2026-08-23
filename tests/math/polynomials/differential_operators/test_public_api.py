"""Native public API contract for differential operators."""

from jacobian.math.polynomials import differential_operators


def test_exact_public_api_symbols() -> None:
    expected = (
        "ConstantCoefficientDifferentialOperator",
        "DifferentialOperatorTerm",
        "apply_constant_coefficient_differential_operator",
    )
    assert tuple(differential_operators.__all__) == expected
    assert len(differential_operators.__all__) == len(
        set(differential_operators.__all__)
    )
    assert all(hasattr(differential_operators, name) for name in expected)
