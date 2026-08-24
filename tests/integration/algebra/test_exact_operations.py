from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.number_field import ring_of_integers
from jacobian.math.number_field._models import NumberFieldRequest
from jacobian.math.number_field._operations import compute_nf_discriminant
from jacobian.math.recurrence_solving._models import (
    ClosedFormRequest,
    RecurrenceFindRequest,
)
from jacobian.math.recurrence_solving._operations import (
    compute_closed_form,
    compute_find_recurrence,
)
from jacobian.math.root_isolation._models import (
    AlgebraicCompareRequest,
    UnivariatePolynomialRequest,
)
from jacobian.math.root_isolation._operations import (
    compute_algebraic_compare,
    compute_root_isolation,
)


def _quadratic(constant: str) -> list[dict[str, str]]:
    return [
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": constant, "den": "1"},
    ]


def _r(value: int) -> CanonicalRational:
    return CanonicalRational(num=str(value), den="1")


def test_root_isolation_returns_intervals_aligned_with_multiplicities() -> None:
    result = compute_root_isolation(
        UnivariatePolynomialRequest.model_validate(
            {"coefficients_descending": _quadratic("-2")}
        )
    )

    assert result.multiplicities == (1, 1)
    assert tuple((left.num, right.num) for left, right in result.roots) == (
        ("-2", "-1"),
        ("1", "2"),
    )


def test_root_isolation_accepts_sympy_singleton_interval_for_a_rational_root() -> None:
    result = compute_root_isolation(
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": "1", "den": "1"},
                    {"num": "-1", "den": "1"},
                ]
            }
        )
    )

    assert tuple(
        (lower.num, lower.den, upper.num, upper.den) for lower, upper in result.roots
    ) == (("1", "1", "1", "1"),)
    assert result.multiplicities == (1,)


def test_algebraic_comparison_parses_canonical_interval_endpoints() -> None:
    result = compute_algebraic_compare(
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": _quadratic("-2"),
                    "isolating_interval_lower": {"num": "1", "den": "1"},
                    "isolating_interval_upper": {"num": "2", "den": "1"},
                },
                "right": {
                    "polynomial": _quadratic("-3"),
                    "isolating_interval_lower": {"num": "1", "den": "1"},
                    "isolating_interval_upper": {"num": "2", "den": "1"},
                },
            }
        )
    )

    assert result.order == "LT"


def test_algebraic_comparison_accepts_coefficients_above_python_digit_limit() -> None:
    oversized_coefficient = "1" + "0" * 5_000
    result = compute_algebraic_compare(
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": [
                        {"num": oversized_coefficient, "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    "isolating_interval_lower": {"num": "-1", "den": "1"},
                    "isolating_interval_upper": {"num": "1", "den": "1"},
                },
                "right": {
                    "polynomial": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    "isolating_interval_lower": {"num": "-1", "den": "1"},
                    "isolating_interval_upper": {"num": "1", "den": "1"},
                },
            }
        )
    )

    assert result.order == "EQ"


def test_algebraic_comparison_contract_rejects_a_nonisolating_interval() -> None:
    with pytest.raises(ValidationError, match="exactly one real root"):
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": _quadratic("-2"),
                    "isolating_interval_lower": {"num": "-2", "den": "1"},
                    "isolating_interval_upper": {"num": "2", "den": "1"},
                },
                "right": {
                    "polynomial": _quadratic("-3"),
                    "isolating_interval_lower": {"num": "1", "den": "1"},
                    "isolating_interval_upper": {"num": "2", "den": "1"},
                },
            }
        )


def test_number_field_discriminant_is_not_power_basis_discriminant() -> None:
    result = compute_nf_discriminant(
        NumberFieldRequest(coefficients_descending=("1", "0", "-5"), variable="x")
    )

    assert result.discriminant == "5"


@pytest.mark.parametrize(
    "variable",
    ["", " ", "x y", "x; y", "x\n", "x\x00", "1x", "x" * 33],
)
def test_number_field_variable_uses_polynomial_identifier_grammar(
    variable: str,
) -> None:
    with pytest.raises(ValidationError):
        NumberFieldRequest(
            coefficients_descending=("1", "0", "-2"),
            variable=variable,
        )


def test_integral_basis_is_computed_in_the_defining_power_basis() -> None:
    assert ring_of_integers(["1", "0", "-5"], "x") == ["1", "x/2 + 1/2"]


def test_number_field_requires_a_monic_irreducible_integer_polynomial() -> None:
    with pytest.raises(ValidationError, match="monic"):
        NumberFieldRequest(coefficients_descending=("2", "0", "-10"), variable="x")


def test_recurrence_finder_solves_for_coefficients() -> None:
    result = compute_find_recurrence(
        RecurrenceFindRequest(sequence=tuple(_r(value) for value in (3, 6, 12, 24)))
    )

    assert result.order == 1
    assert result.coefficients == (_r(2),)


def test_native_recurrence_api_enforces_the_sequence_contract() -> None:
    from jacobian.math.recurrence_solving import closed_form, find_recurrence

    recurrence = find_recurrence(tuple(_r(value) for value in (3, 6, 12, 24)))
    assert recurrence.status == "FOUND"
    assert recurrence.coefficients == (_r(2),)
    assert closed_form((_r(1), _r(-2)), (_r(3),)).expression == "3*2**n"

    with pytest.raises(ValidationError, match="at least 2"):
        find_recurrence((_r(1),))

    with pytest.raises(ValidationError, match="initial value count"):
        closed_form((_r(1), _r(-1), _r(-1)), (_r(1),))


def test_recurrence_finder_reports_a_missing_nonvacuous_fit() -> None:
    result = compute_find_recurrence(RecurrenceFindRequest(sequence=(_r(0), _r(1))))

    assert result.status == "NO_FITTING_RECURRENCE"
    assert result.order == 0
    assert result.coefficients == ()


def test_repeated_root_closed_form_preserves_polynomial_factor() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=(_r(1), _r(-2), _r(1)),
            initial_values=(_r(2), _r(5)),
        )
    )

    assert result.expression == "3*n + 2"


def test_closed_form_handles_repeated_zero_characteristic_roots() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=(_r(1), _r(0), _r(0)),
            initial_values=(_r(2), _r(5)),
        )
    )

    assert result.expression == "2*KroneckerDelta(0, n) + 5*KroneckerDelta(1, n)"


def test_closed_form_contract_rejects_characteristic_polynomials_above_degree_four() -> (
    None
):
    with pytest.raises(ValidationError):
        ClosedFormRequest(
            characteristic_coefficients=tuple(
                _r(value) for value in (1,) + (0,) * 16 + (-1,)
            ),
            initial_values=tuple(_r(0) for _ in range(17)),
        )


def test_closed_form_contract_requires_every_initial_value() -> None:
    with pytest.raises(ValidationError, match="initial value count"):
        ClosedFormRequest(
            characteristic_coefficients=(_r(1), _r(-1), _r(-1)),
            initial_values=(_r(1),),
        )
