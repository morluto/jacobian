"""Native and wire parity for P-recursive table residuals."""

from fractions import Fraction

from jacobian.math.combinatorics.recurrence_tables import (
    PolynomialCoefficientRecurrenceTableRequest,
    _compute_recurrence_table_residuals,
    recurrence_table_residuals,
)


def test_native_recurrence_table_residuals_accepts_canonical_rationals() -> None:
    """Native callers pass values, while MCP retains its strict wire request."""

    coefficients = ((Fraction(1),), (Fraction(), Fraction(-1)))
    values = tuple(Fraction(value) for value in (1, 1, 2, 6, 24, 120))

    native = recurrence_table_residuals(coefficients, values)
    wire = _compute_recurrence_table_residuals(
        PolynomialCoefficientRecurrenceTableRequest.model_validate(
            {
                "coefficient_polynomials": [
                    [{"num": "1", "den": "1"}],
                    [{"num": "0", "den": "1"}, {"num": "-1", "den": "1"}],
                ],
                "values": [
                    {"num": str(value), "den": "1"} for value in (1, 1, 2, 6, 24, 120)
                ],
                "coefficient_convention": (
                    "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
                ),
                "polynomial_convention": "ASCENDING_POWERS_OF_N",
                "table_convention": "VALUES_A_0_THROUGH_A_N_IN_ORDER",
            }
        )
    )

    assert native == wire
    assert native.satisfies_recurrence is True
    assert all(residual.value.as_fraction() == 0 for residual in native.residuals)
