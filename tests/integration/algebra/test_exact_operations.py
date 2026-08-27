from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError
from tests.integration.algebra._support import algebra_validation_error

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
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
    MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS,
    AlgebraicCompareRequest,
    UnivariatePolynomialRequest,
)
from jacobian.math.root_isolation._operations import (
    _verify_root_isolation_result,
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


def test_root_isolation_returns_source_bound_composable_identities() -> None:
    request = UnivariatePolynomialRequest.model_validate(
        {"coefficients_descending": _quadratic("-2")}
    )
    result = compute_root_isolation(request)

    assert result.source_coefficients_descending == ("1", "0", "-2")
    assert tuple(
        (entry.isolating_interval[0].num, entry.isolating_interval[1].num)
        for entry in result.roots
    ) == (
        ("-2", "-1"),
        ("1", "2"),
    )
    assert tuple(entry.algebraic_value.real_root_index for entry in result.roots) == (
        0,
        1,
    )
    assert type(result).model_validate(result.model_dump(mode="json")) == result
    assert _verify_root_isolation_result(result)

    forged = result.model_copy(
        update={
            "roots": (
                result.roots[0].model_copy(update={"multiplicity": 2}),
                result.roots[1],
            )
        }
    )
    assert not _verify_root_isolation_result(forged)


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

    root = result.roots[0]
    lower, upper = root.isolating_interval
    assert (lower.num, lower.den, upper.num, upper.den) == ("1", "1", "1", "1")
    assert root.multiplicity == 1
    assert root.algebraic_value.polynomial == ("1", "-1")


def test_root_isolation_preserves_factor_identity_and_source_multiplicity() -> None:
    result = compute_root_isolation(
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "-5", "den": "1"},
                    {"num": "-1", "den": "1"},
                    {"num": "8", "den": "1"},
                    {"num": "-4", "den": "1"},
                ]
            }
        )
    )

    # (x - 1)^3 (x + 2)^2 has degree five and two distinct real roots.
    assert tuple(root.multiplicity for root in result.roots) == (2, 3)
    assert tuple(root.algebraic_value.polynomial for root in result.roots) == (
        ("1", "2"),
        ("1", "-1"),
    )
    comparison = compute_algebraic_compare(
        AlgebraicCompareRequest(
            left=result.roots[0].algebraic_value,
            right=result.roots[1].algebraic_value,
        )
    )
    assert comparison.order == "LT"


def test_root_isolation_keeps_an_empty_source_bound_real_root_family() -> None:
    result = compute_root_isolation(
        UnivariatePolynomialRequest.model_validate(
            {"coefficients_descending": _quadratic("1")}
        )
    )

    assert result.source_coefficients_descending == ("1", "0", "1")
    assert result.roots == ()
    assert _verify_root_isolation_result(result)


def test_root_isolation_normalizes_rational_source_before_identity_projection() -> None:
    result = compute_root_isolation(
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": "2", "den": "1"},
                    {"num": "-1", "den": "1"},
                ]
            }
        )
    )

    assert result.source_coefficients_descending == ("2", "-1")
    assert result.roots[0].algebraic_value.polynomial == ("2", "-1")


def test_root_isolation_rejects_sources_outside_the_composable_envelope() -> None:
    with pytest.raises(ValidationError):
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": "1", "den": "1"},
                    *({"num": "0", "den": "1"} for _ in range(9)),
                ]
            }
        )
    with pytest.raises(ValidationError):
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": "1" + "0" * 996, "den": "1"},
                    {"num": "1", "den": "1"},
                ]
            }
        )


def test_root_isolation_rejects_expanded_normalization_without_decimal_formatting() -> (
    None
):
    """Clearing valid rational denominators must return the intended bound error."""

    base = 10 ** (MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS - 1)
    with pytest.raises(ValidationError, match="source_coefficient_bound"):
        UnivariatePolynomialRequest.model_validate(
            {
                "coefficients_descending": [
                    {"num": str(base + 1), "den": str(base - 1)},
                    {"num": "1", "den": str(base + 4)},
                ]
            }
        )


def test_algebraic_comparison_parses_canonical_interval_endpoints() -> None:
    result = compute_algebraic_compare(
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": ["1", "0", "-2"],
                    "real_root_index": 1,
                },
                "right": {
                    "polynomial": ["1", "0", "-3"],
                    "real_root_index": 1,
                },
            }
        )
    )

    assert result.order == "LT"
    assert result.left_isolating_interval.lower.as_fraction() == 1
    assert result.left_isolating_interval.upper.as_fraction() < 2
    assert result.right_isolating_interval.lower.as_fraction() == Fraction(3, 2)
    assert result.right_isolating_interval.upper.as_fraction() == 2


def test_algebraic_comparison_rejects_coefficients_above_its_work_bound() -> None:
    oversized_coefficient = "1" + "0" * 1_000
    with algebra_validation_error():
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": [oversized_coefficient, "1"],
                    "real_root_index": 0,
                },
                "right": {
                    "polynomial": ["1", "0"],
                    "real_root_index": 0,
                },
            }
        )


def test_algebraic_comparison_contract_rejects_a_missing_real_root() -> None:
    with algebra_validation_error():
        AlgebraicCompareRequest.model_validate(
            {
                "left": {
                    "polynomial": ["1", "0", "-2"],
                    "real_root_index": 2,
                },
                "right": {
                    "polynomial": ["1", "0", "-3"],
                    "real_root_index": 1,
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
    with algebra_validation_error():
        NumberFieldRequest(coefficients_descending=("2", "0", "-10"), variable="x")


def test_number_field_reducibility_is_an_owner_declared_invalid_request() -> None:
    request = NumberFieldRequest(coefficients_descending=("1", "0", "-1"), variable="x")

    with pytest.raises(OperationDomainValidationError) as caught:
        compute_nf_discriminant(request)

    assert caught.value.errors()[0]["type"] == "number_field.not_irreducible"


def test_number_field_rejects_oversized_coefficients_before_sympy() -> None:
    with pytest.raises(ValidationError) as caught:
        NumberFieldRequest(
            coefficients_descending=("1" + "0" * 256, "0", "-2"),
            variable="x",
        )

    assert caught.value.errors()[0]["type"] == "number_field.coefficient_digits"


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

    with algebra_validation_error():
        find_recurrence((_r(1),))

    with algebra_validation_error():
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
    with algebra_validation_error():
        ClosedFormRequest(
            characteristic_coefficients=(_r(1), _r(-1), _r(-1)),
            initial_values=(_r(1),),
        )
