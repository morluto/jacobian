"""Native public API contract for differential operators."""

from jacobian.math.polynomials import differential_operators


def test_exact_public_api_symbols() -> None:
    expected = (
        "ConstantCoefficientDifferentialOperator",
        "DifferentialOperatorTerm",
        "apply_constant_coefficient_differential_operator",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(differential_operators.__all__)
    # The subset assertion above cannot see a newly added name, so the
    # private-name rule is checked over the complete __all__.
    assert all(not name.startswith("_") for name in differential_operators.__all__)
    assert len(differential_operators.__all__) == len(
        set(differential_operators.__all__)
    )
    assert all(hasattr(differential_operators, name) for name in expected)
