from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.number_field import NumberFieldRequest
from jacobian.contracts.recurrence_solving import (
    ClosedFormRequest,
    RecurrenceFindRequest,
)
from jacobian.contracts.root_isolation import (
    AlgebraicCompareRequest,
    UnivariatePolynomialRequest,
)
from jacobian.domains.number_field.operations import compute_nf_discriminant
from jacobian.domains.recurrence_solving.operations import (
    compute_closed_form,
    compute_find_recurrence,
)
from jacobian.domains.root_isolation.operations import (
    compute_algebraic_compare,
    compute_root_isolation,
)
from jacobian.math.number_field import ring_of_integers


def _quadratic(constant: str) -> list[dict[str, str]]:
    return [
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": constant, "den": "1"},
    ]


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


def test_integral_basis_is_computed_in_the_defining_power_basis() -> None:
    assert ring_of_integers(["1", "0", "-5"], "x") == ["1", "x/2 + 1/2"]


def test_number_field_requires_a_monic_irreducible_integer_polynomial() -> None:
    with pytest.raises(ValidationError, match="monic"):
        NumberFieldRequest(coefficients_descending=("2", "0", "-10"), variable="x")


def test_recurrence_finder_solves_for_coefficients() -> None:
    result = compute_find_recurrence(
        RecurrenceFindRequest(sequence=("3", "6", "12", "24"))
    )

    assert result.order == 1
    assert result.coefficients == ("2",)


def test_repeated_root_closed_form_preserves_polynomial_factor() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=("1", "-2", "1"),
            initial_values=("2", "5"),
        )
    )

    assert result.expression == "3*n + 2"


def test_closed_form_contract_requires_every_initial_value() -> None:
    with pytest.raises(ValidationError, match="initial value count"):
        ClosedFormRequest(
            characteristic_coefficients=("1", "-1", "-1"), initial_values=("1",)
        )
