from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction

import pytest
from pydantic import BaseModel, ValidationError
from tests.integration.algebra._support import algebra_validation_error

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers import real as real_algebraic
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS,
    AlgebraicCompareRequest,
    UnivariatePolynomialRequest,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._sympy import (
    compute_algebraic_compare,
    compute_root_isolation,
)
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
    discriminant,
    embeddings,
    ring_of_integers,
)
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_DEGREE,
)
from jacobian.math.number_theory.sequences.recurrence_solving._models import (
    ClosedFormRequest,
    RecurrenceFindRequest,
)
from jacobian.math.number_theory.sequences.recurrence_solving._tools import (
    compute_closed_form,
    compute_find_recurrence,
)


def _quadratic(constant: str) -> list[dict[str, str]]:
    return [
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": constant, "den": "1"},
    ]


def _r(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _wire[ModelT: BaseModel](model: type[ModelT], payload: object) -> ModelT:
    return model.model_validate_json(json.dumps(payload))


def test_root_isolation_returns_source_bound_composable_identities() -> None:
    request = _wire(UnivariatePolynomialRequest, _isolation_payload(_quadratic("-2")))
    result = compute_root_isolation(request)

    assert result.source_polynomial == request.polynomial
    assert tuple(
        (entry.isolating_interval[0].num, entry.isolating_interval[1].num)
        for entry in result.roots
    ) == (
        (-2, -1),
        (1, 2),
    )
    assert tuple(entry.algebraic_value.real_root_index for entry in result.roots) == (
        0,
        1,
    )
    assert (
        type(result).model_validate_json(json.dumps(result.model_dump(mode="json")))
        == result
    )

    forged = result.model_copy(
        update={
            "roots": (
                result.roots[0].model_copy(update={"multiplicity": 2}),
                result.roots[1],
            )
        }
    )
    assert forged.roots[0].multiplicity == 2


def test_root_isolation_uses_admitted_root_carriers_without_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_replayed(*_args: int) -> int:
        raise AssertionError("root carrier gcd validation was replayed")

    monkeypatch.setattr(real_algebraic, "gcd", fail_if_replayed)

    result = compute_root_isolation(
        _wire(UnivariatePolynomialRequest, _isolation_payload(_quadratic("-2")))
    )

    assert tuple(root.algebraic_value.real_root_index for root in result.roots) == (
        0,
        1,
    )


def test_root_isolation_accepts_sympy_singleton_interval_for_a_rational_root() -> None:
    result = compute_root_isolation(
        _wire(
            UnivariatePolynomialRequest,
            _isolation_payload(
                [
                    {"num": "1", "den": "1"},
                    {"num": "-1", "den": "1"},
                ]
            ),
        )
    )

    root = result.roots[0]
    lower, upper = root.isolating_interval
    assert (lower.num, lower.den, upper.num, upper.den) == (1, 1, 1, 1)
    assert root.multiplicity == 1
    assert root.algebraic_value.polynomial == (1, -1)


def test_root_isolation_preserves_factor_identity_and_source_multiplicity() -> None:
    result = compute_root_isolation(
        _wire(
            UnivariatePolynomialRequest,
            _isolation_payload(
                [
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "-5", "den": "1"},
                    {"num": "-1", "den": "1"},
                    {"num": "8", "den": "1"},
                    {"num": "-4", "den": "1"},
                ]
            ),
        )
    )

    # (x - 1)^3 (x + 2)^2 has degree five and two distinct real roots.
    assert tuple(root.multiplicity for root in result.roots) == (2, 3)
    assert tuple(root.algebraic_value.polynomial for root in result.roots) == (
        (1, 2),
        (1, -1),
    )
    comparison = compute_algebraic_compare(
        AlgebraicCompareRequest(
            left=result.roots[0].algebraic_value,
            right=result.roots[1].algebraic_value,
        )
    )
    assert comparison.order == "LT"


def test_root_isolation_keeps_an_empty_source_bound_real_root_family() -> None:
    request = _wire(UnivariatePolynomialRequest, _isolation_payload(_quadratic("1")))
    result = compute_root_isolation(request)

    assert result.source_polynomial == request.polynomial
    assert result.roots == ()


def test_root_isolation_retains_rational_source_before_identity_projection() -> None:
    request = _wire(
        UnivariatePolynomialRequest,
        _isolation_payload(
            [
                {"num": "2", "den": "1"},
                {"num": "-1", "den": "1"},
            ]
        ),
    )
    result = compute_root_isolation(request)

    assert result.source_polynomial == request.polynomial
    assert result.roots[0].algebraic_value.polynomial == (2, -1)


def test_root_isolation_rejects_sources_outside_the_composable_envelope() -> None:
    with pytest.raises(ValidationError):
        _wire(
            UnivariatePolynomialRequest,
            _isolation_payload(
                [
                    {"num": "1", "den": "1"},
                    *({"num": "0", "den": "1"} for _ in range(9)),
                ]
            ),
        )
    with pytest.raises(ValidationError):
        _wire(
            UnivariatePolynomialRequest,
            _isolation_payload(
                [
                    {"num": "1" + "0" * 996, "den": "1"},
                    {"num": "1", "den": "1"},
                ]
            ),
        )


def test_root_isolation_rejects_expanded_normalization_without_decimal_formatting() -> (
    None
):
    """Clearing valid rational denominators must return the intended bound error."""

    base = 10 ** (MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS - 1)
    with pytest.raises(ValueError, match="primitive integer source coefficients"):
        compute_root_isolation(
            _wire(
                UnivariatePolynomialRequest,
                _isolation_payload(
                    [
                        {"num": str(base + 1), "den": str(base - 1)},
                        {"num": "1", "den": str(base + 4)},
                    ]
                ),
            )
        )


def test_algebraic_comparison_parses_canonical_interval_endpoints() -> None:
    result = compute_algebraic_compare(
        _wire(
            AlgebraicCompareRequest,
            {
                "left": {
                    "polynomial": ["1", "0", "-2"],
                    "real_root_index": 1,
                },
                "right": {
                    "polynomial": ["1", "0", "-3"],
                    "real_root_index": 1,
                },
            },
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
        _wire(
            AlgebraicCompareRequest,
            {
                "left": {
                    "polynomial": [oversized_coefficient, "1"],
                    "real_root_index": 0,
                },
                "right": {
                    "polynomial": ["1", "0"],
                    "real_root_index": 0,
                },
            },
        )


def test_algebraic_comparison_contract_rejects_a_missing_real_root() -> None:
    with algebra_validation_error():
        _wire(
            AlgebraicCompareRequest,
            {
                "left": {
                    "polynomial": ["1", "0", "-2"],
                    "real_root_index": 2,
                },
                "right": {
                    "polynomial": ["1", "0", "-3"],
                    "real_root_index": 1,
                },
            },
        )


def test_number_field_discriminant_is_not_power_basis_discriminant() -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    result = compute_nf_discriminant(NumberFieldRequest(field=field))

    assert result.discriminant == 5


def test_embedding_field_composes_unchanged_with_field_invariant_consumers() -> None:
    produced = embeddings(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, 1))
    ).field
    request = _wire(NumberFieldRequest, {"field": produced.model_dump(mode="json")})

    assert request.field == produced
    assert compute_nf_discriminant(request).discriminant == -4
    assert discriminant(request.field) == -4
    assert ring_of_integers(request.field) == ["1", "alpha"]


def test_field_discriminant_request_schema_uses_the_canonical_presentation() -> None:
    request_schema = NumberFieldRequest.model_json_schema()
    field_reference = request_schema["properties"]["field"]["$ref"]
    field_schema = request_schema["$defs"][field_reference.rsplit("/", 1)[-1]]

    assert field_schema["title"] == "SimpleNumberFieldPresentation"
    assert set(field_schema["properties"]) == {
        "domain",
        "coefficients_descending",
    }
    assert (
        field_schema["properties"]["coefficients_descending"]["maxItems"]
        == MAX_SIMPLE_NUMBER_FIELD_DEGREE + 1
    )


def test_field_carrier_preserves_the_prior_discriminant_degree_envelope() -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 30, 1))

    assert field.degree == 31
    assert NumberFieldRequest(field=field).field is field


@pytest.mark.parametrize("consumer", (discriminant, ring_of_integers))
def test_native_integral_basis_consumers_preserve_degree_31_envelope(
    consumer: Callable[[SimpleNumberFieldPresentation], object],
) -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 30, -2))

    result = consumer(field)

    if consumer is discriminant:
        assert result == -18327886165296381817380980351835033630345588173537542144
    else:
        assert result == [
            "1",
            "alpha",
            *[f"alpha**{power}" for power in range(2, field.degree)],
        ]


@pytest.mark.parametrize("consumer", (discriminant, ring_of_integers))
def test_native_integral_basis_consumers_accept_degree_nine_field(
    consumer: Callable[[SimpleNumberFieldPresentation], object],
) -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 8, -2))

    result = consumer(field)

    if consumer is discriminant:
        assert result == 99179645184
    else:
        assert result == [
            "1",
            "alpha",
            *[f"alpha**{power}" for power in range(2, field.degree)],
        ]


@pytest.mark.parametrize("consumer", (discriminant, ring_of_integers))
def test_native_integral_basis_consumers_bound_the_widened_field_carrier(
    consumer: Callable[[SimpleNumberFieldPresentation], object],
) -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 31, -2))

    with pytest.raises(ValueError, match="limited to degree 31"):
        consumer(field)


def test_integral_basis_is_computed_in_the_defining_power_basis() -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))

    assert ring_of_integers(field) == ["1", "alpha/2 + 1/2"]
    assert discriminant(field) == 5


def test_number_field_consumers_accept_a_nonmonic_canonical_presentation() -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(2, 0, 1))

    assert compute_nf_discriminant(NumberFieldRequest(field=field)).discriminant == -8
    assert discriminant(field) == -8
    assert ring_of_integers(field) == ["1", "2*alpha"]


def test_number_field_reducibility_is_an_owner_declared_invalid_request() -> None:
    request = _wire(
        NumberFieldRequest,
        {
            "field": {
                "domain": "QQ",
                "coefficients_descending": ["1", "0", "-1"],
            }
        },
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        compute_nf_discriminant(request)

    assert caught.value.errors()[0]["type"] == "number_field.not_irreducible"


def test_number_field_rejects_oversized_coefficients_before_sympy() -> None:
    with pytest.raises(ValidationError) as caught:
        _wire(
            NumberFieldRequest,
            {
                "field": {
                    "domain": "QQ",
                    "coefficients_descending": ["1" + "0" * 256, "0", "-2"],
                }
            },
        )

    assert caught.value.errors()[0]["type"] == "simple_number_field.coefficient_bound"


def test_recurrence_finder_solves_for_coefficients() -> None:
    result = compute_find_recurrence(
        RecurrenceFindRequest(sequence=tuple(_r(value) for value in (3, 6, 12, 24)))
    )

    assert result.order == 1
    assert result.coefficients == (_r(2),)


def test_native_recurrence_api_enforces_the_sequence_contract() -> None:
    from jacobian.math.number_theory.sequences.recurrence_solving import (
        closed_form,
        find_recurrence,
    )

    recurrence = find_recurrence(tuple(_r(value) for value in (3, 6, 12, 24)))
    assert recurrence.status == "FOUND"
    assert recurrence.coefficients == (_r(2),)
    assert closed_form((_r(1), _r(-2)), (_r(3),)).expression == "3*2**n"

    with pytest.raises(ValueError, match="sequence must have length"):
        find_recurrence((_r(1),))

    with pytest.raises(ValueError, match="initial value count"):
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

    assert result.expression.value == "3*n + 2"


def test_closed_form_handles_repeated_zero_characteristic_roots() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=(_r(1), _r(0), _r(0)),
            initial_values=(_r(2), _r(5)),
        )
    )

    assert result.expression.value == "2*KroneckerDelta(0, n) + 5*KroneckerDelta(1, n)"


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
    request = ClosedFormRequest(
        characteristic_coefficients=(_r(1), _r(-1), _r(-1)),
        initial_values=(_r(1),),
    )
    with pytest.raises(OperationDomainValidationError, match="initial value count"):
        compute_closed_form(request)


def _isolation_payload(coefficients: list[dict[str, str]]) -> dict[str, object]:
    return {
        "polynomial": {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": value, "exponents": [len(coefficients) - 1 - i]}
                    for i, value in enumerate(coefficients)
                    if value["num"] != "0"
                ]
            },
        }
    }
