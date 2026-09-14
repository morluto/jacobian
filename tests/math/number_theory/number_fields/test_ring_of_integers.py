"""Ring-of-integers bases for presented simple number fields (#1667)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import pytest
import sympy
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

import jacobian.math.number_theory.number_fields._integral_basis as integral_basis_kernel
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields._integral_basis import (
    monicized_discriminant_digit_bound,
    recognized_integral_basis,
    require_factorizable_discriminant,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldRingOfIntegersRequest,
)
from jacobian.math.number_theory.number_fields._ring_of_integers import (
    NumberFieldRingOfIntegersResult,
    ring_of_integers,
)
from jacobian.math.number_theory.number_fields._tools import (
    compute_ring_of_integers,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS,
    MAX_SIMPLE_NUMBER_FIELD_DEGREE,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)


def _coordinates(result: NumberFieldRingOfIntegersResult) -> list[list[Fraction]]:
    return [
        [coefficient.as_fraction() for coefficient in element.coefficients_ascending]
        for element in result.basis
    ]


def _cyclotomic_coefficients(order: int) -> tuple[int, ...]:
    symbol = sympy.Symbol("alpha")
    polynomial = sympy.cyclotomic_poly(order, symbol)
    return tuple(
        int(coefficient)
        for coefficient in sympy.Poly(polynomial, symbol, domain=sympy.ZZ).all_coeffs()
    )


def test_golden_field_is_not_the_naive_power_basis() -> None:
    """Q(sqrt(5)) has integral basis 1 and (1+sqrt(5))/2, so halve coordinates."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    assert result.field_discriminant == 5
    assert _coordinates(result) == [
        [Fraction(1), Fraction(0)],
        [Fraction(1, 2), Fraction(1, 2)],
    ]
    assert all(
        isinstance(element, SimpleNumberFieldElement) for element in result.basis
    )
    assert all(element.presentation is result.field for element in result.basis)


def test_squarefree_discriminant_field_keeps_the_power_basis() -> None:
    """Q(sqrt(2)) is already integrally closed, so its basis is 1, sqrt(2)."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -2))
    )
    assert result.field_discriminant == 8
    assert _coordinates(result) == [
        [Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(1)],
    ]


@pytest.mark.parametrize(
    ("order", "discriminant"),
    ((3, -3), (4, -4), (5, 125), (7, -16807)),
)
def test_cyclotomic_fields_have_full_power_basis(order: int, discriminant: int) -> None:
    """A cyclotomic field of prime order is monogenic with 1, zeta, ..., zeta^n."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(
            coefficients_descending=_cyclotomic_coefficients(order)
        )
    )
    degree = result.field.degree
    assert result.field_discriminant == discriminant
    assert _coordinates(result) == [
        [Fraction(1 if row == column else 0) for column in range(degree)]
        for row in range(degree)
    ]


def test_basis_vectors_are_bounded_rationals_of_full_degree() -> None:
    """Every returned vector spans the complete power basis with exact rationals."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, 0, 0, -2))
    )
    assert len(result.basis) == result.field.degree
    assert all(
        len(element.coefficients_ascending) == result.field.degree
        for element in result.basis
    )


def test_reducible_polynomial_is_rejected_without_a_basis() -> None:
    """A reducible defining polynomial does not present a number field."""
    # (x^2 - 2)(x^2 - 3) is reducible over QQ while remaining primitive.
    with pytest.raises(OperationDomainValidationError) as error:
        ring_of_integers(
            SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5, 0, 6))
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.defining_polynomial_must_be_irreducible"
    )


def test_ring_of_integers_round_trips_through_strict_json() -> None:
    """The declared basis survives strict JSON serialization unchanged."""
    result = compute_ring_of_integers(
        NumberFieldRingOfIntegersRequest(
            field=SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
        )
    )
    restored = NumberFieldRingOfIntegersResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_ring_of_integers_request_schema_and_parser_share_degree_boundary() -> None:
    input_schema = NumberFieldRingOfIntegersRequest.model_json_schema(mode="validation")
    field_schema = input_schema["properties"]["field"]
    assert field_schema["properties"]["coefficients_descending"]["maxItems"] == 32

    schema_validator = Draft202012Validator(input_schema)
    schema_validator.validate(
        {"field": {"coefficients_descending": ["1", *(["0"] * 30), "-2"]}}
    )
    with pytest.raises(JSONSchemaValidationError):
        schema_validator.validate(
            {"field": {"coefficients_descending": ["1", *(["0"] * 31), "-2"]}}
        )

    accepted = NumberFieldRingOfIntegersRequest(
        field=SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 30, -2))
    )
    assert accepted.field.degree == 31

    with pytest.raises(ValidationError, match="degree at most 31"):
        NumberFieldRingOfIntegersRequest.model_validate(
            {"field": {"coefficients_descending": (1, *(0,) * 31, -2)}}
        )


def test_package_entry_point_uses_the_typed_implementation() -> None:
    """Native package callers receive the composable typed result."""

    from jacobian.math.number_theory import number_fields
    from jacobian.math.number_theory.number_fields import _ring_of_integers

    assert number_fields.ring_of_integers is _ring_of_integers.ring_of_integers
    result = number_fields.ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    assert isinstance(result, NumberFieldRingOfIntegersResult)
    assert result.field_discriminant == 5


def test_semiprime_discriminant_is_rejected_before_backend_expansion() -> None:
    """A semiprime discriminant is not factorable, so admission rejects it.

    ``x^2 - N`` with ``N`` a product of two primes above the trial envelope
    fits the coefficient carrier, but round_two would have to factor the
    discriminant ``4N``. Both the native and process entry points must raise
    the typed admission error instead of launching unbounded work.
    """

    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, 0, -100003 * 100019)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        ring_of_integers(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )


def test_semiprime_discriminant_is_rejected_inside_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discriminant admission algebra runs inside the killable worker."""
    from jacobian import process as process_runtime
    from jacobian.math.number_theory.number_fields._ring_of_integers_process import (
        compute_nf_ring_of_integers,
    )

    launched = False
    original = process_runtime.run_bounded_process

    def record_launch(
        *args: Any, **kwargs: Any
    ) -> process_runtime.BoundedProcessResult:
        nonlocal launched
        launched = True
        return original(*args, **kwargs)

    monkeypatch.setattr(process_runtime, "run_bounded_process", record_launch)
    request = NumberFieldRingOfIntegersRequest(
        field=SimpleNumberFieldPresentation(
            coefficients_descending=(1, 0, -100003 * 100019)
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_nf_ring_of_integers(request)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )
    assert launched


def test_discriminant_admission_does_not_factor_inside_perfect_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_factoring(candidate: object, factor: bool = True) -> bool:
        assert factor is False
        return False

    monkeypatch.setattr(
        "sympy.ntheory.perfect_power",
        fail_if_factoring,
    )
    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, 0, -100003 * 100019)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        ring_of_integers(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )


def test_admitted_monic_discriminant_is_request_scoped_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    admitted = require_factorizable_discriminant(field)
    assert admitted == 20
    assert not hasattr(integral_basis_kernel, "_ADMITTED_MONIC_DISCRIMINANTS")

    def fail_if_poly_discriminant_rerun(
        self: object, *args: object, **kwargs: object
    ) -> object:
        raise AssertionError(
            "admitted discriminant must not recompute Poly.discriminant"
        )

    def fail_if_irreducible_rerun(self: object) -> bool:
        raise AssertionError(
            "admitted irreducibility must not recompute Poly.is_irreducible"
        )

    monkeypatch.setattr(sympy.Poly, "discriminant", fail_if_poly_discriminant_rerun)
    monkeypatch.setattr(
        sympy.Poly, "is_irreducible", property(fail_if_irreducible_rerun)
    )
    recognized = recognized_integral_basis(
        field,
        admitted_polynomial_discriminant=admitted,
        admitted_irreducible=True,
    )
    assert recognized is not None
    _ring, field_discriminant, _alpha, _leading = recognized
    assert int(field_discriminant) == 5


def test_admitted_discriminant_seeds_the_round_two_factor_cache() -> None:
    """Admission hands the proved cofactor to SymPy's factor cache.

    ``round_two`` factors the same discriminant again; recording the proved
    prime for the post-trial remainder lets the backend skip the repeated
    primality test instead of replaying the admission mathematics.
    """

    from sympy import factor_cache
    from sympy.ntheory.factor_ import _factorint_small

    prime = int(sympy.nextprime(10**200))
    while prime % 4 != 1:
        prime = int(sympy.nextprime(prime + 1))
    field = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -prime))
    factor_cache.clear()
    discriminant = require_factorizable_discriminant(field)
    assert discriminant is not None
    remainder, _ = _factorint_small({}, abs(discriminant), 2**15, 600)
    assert factor_cache.get(int(remainder)) == prime


def test_reducible_semiprime_linear_factor_is_not_a_factorization_bound() -> None:
    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, -(100003 * 100019), 0)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        ring_of_integers(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.defining_polynomial_must_be_irreducible"
    )


def test_nonmonic_eisenstein_discriminant_is_rejected_before_worker_launch() -> None:
    field = SimpleNumberFieldPresentation(
        coefficients_descending=(2**849, *([0] * 30), -3)
    )
    assert (
        monicized_discriminant_digit_bound(field) > MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        ring_of_integers(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_output_bound"
    )


def test_mersenne_power_discriminant_rejects_without_converting_the_cofactor() -> None:
    """A Mersenne power discriminant is larger than CPython's str() digit cap.

    ``x^31 - (2^607 - 1)`` has a cofactor of about 5,500 decimal digits, so
    ``len(str(cofactor))`` raises ``ValueError``. Admission must still return
    the typed 4,096-digit bound.
    """

    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, *([0] * 30), -(2**607 - 1))
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        ring_of_integers(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )


def test_ring_of_integers_result_rejects_malformed_basis_on_deserialization() -> None:
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    payload = result.model_dump(mode="json")
    payload["basis"] = payload["basis"][:1]
    with pytest.raises(ValidationError) as error:
        NumberFieldRingOfIntegersResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_basis_length"
    )


def test_ring_of_integers_result_rejects_a_basis_from_another_field() -> None:
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    payload = result.model_dump(mode="json")
    payload["basis"][1]["presentation"]["coefficients_descending"] = [
        "1",
        "0",
        "-2",
    ]

    with pytest.raises(ValidationError) as error:
        NumberFieldRingOfIntegersResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_basis_field"
    )


def test_native_ring_of_integers_runs_in_the_killable_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package-exported native entry shares the catalog process boundary.

    A native Python call is an admitted boundary like ``math.run``: the
    irreducibility, discriminant, and ``round_two`` work must sit behind the
    request-owned killable worker so cancellation and the request envelope are
    observed identically on both routes.
    """

    import jacobian.process as process_runtime

    launched: list[bool] = []
    original = process_runtime.run_bounded_process

    def record_launch(
        *args: Any, **kwargs: Any
    ) -> process_runtime.BoundedProcessResult:
        launched.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(process_runtime, "run_bounded_process", record_launch)
    with pytest.raises(OperationResourceAdmissionError) as error:
        ring_of_integers(
            SimpleNumberFieldPresentation(
                coefficients_descending=(1, 0, -100003 * 100019)
            )
        )
    assert launched
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )


def test_monic_leading_coefficient_adds_no_monicization_growth() -> None:
    """Multiplying by ``1**k`` adds no digits, so monic fields are not charged.

    The previous estimate treated a unit leading coefficient as one digit of
    growth per power and overcharged every coefficient by ``index - 1`` digits,
    rejecting monic fields whose discriminant already fit the carrier.
    """

    unit_leading = SimpleNumberFieldPresentation(
        coefficients_descending=(1, *([0] * 125), -(2 * 3**534))
    )
    assert unit_leading.degree == MAX_SIMPLE_NUMBER_FIELD_DEGREE
    assert monicized_discriminant_digit_bound(unit_leading) == (
        MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS
    )

    # A leading coefficient of ten contributes ceil(log10 10) == 1 digit per
    # power, not the two digits of its own decimal representation, so the
    # constant term is scaled to 10**4 and the envelope grows by four digits.
    ten_leading = SimpleNumberFieldPresentation(
        coefficients_descending=(10, *([0] * 4), -1)
    )
    monic_quintic = SimpleNumberFieldPresentation(
        coefficients_descending=(1, *([0] * 4), -1)
    )
    assert monicized_discriminant_digit_bound(monic_quintic) == 9 * 1 + 4 * 5
    assert monicized_discriminant_digit_bound(ten_leading) == 9 * 5 + 4 * 5


def test_high_degree_monic_field_is_not_rejected_on_the_phantom_digit_estimate() -> (
    None
):
    """The degree-126 Eisenstein discriminant fits the declared carrier.

    ``x^126 + 2*3^534`` has a 32,151-digit polynomial discriminant. The
    monicization envelope must admit it instead of rejecting the request on an
    estimate that charged 125 phantom digits to the constant coefficient.
    """

    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, *([0] * 125), -(2 * 3**534))
    )
    admitted = require_factorizable_discriminant(field)
    assert admitted is not None
    assert abs(admitted) < 10**MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS


def test_native_discriminant_shares_the_worker_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exported `discriminant` must not take a cheaper route than `math.run`.

    It previously called `recognized_integral_basis` synchronously with no
    `require_factorizable_discriminant`, so `x^2 - 100003*100019` was refused on
    the catalog path while the native path entered the unbounded factorization.
    """

    import jacobian.process as process_runtime
    from jacobian.math.number_theory.number_fields.operations import discriminant

    launched: list[bool] = []
    original = process_runtime.run_bounded_process

    def record_launch(*args: Any, **kwargs: Any) -> Any:
        launched.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(process_runtime, "run_bounded_process", record_launch)
    field = SimpleNumberFieldPresentation(
        coefficients_descending=(1, 0, -100003 * 100019)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        discriminant(field)
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )
    assert launched
    # The admitted case still agrees with the catalog adapter.
    assert (
        discriminant(SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5)))
        == 5
    )


def test_native_fields_reject_non_presentation_arguments() -> None:
    """Public native entries validate the runtime type before dereferencing it.

    `ring_of_integers({"coefficients_descending": (1, 0, -5)})` used to raise a
    raw `AttributeError` from `field.degree` instead of the structured domain
    diagnostic this admission boundary owes its callers.
    """

    from jacobian.math.number_theory.number_fields.operations import discriminant

    malformed = {"coefficients_descending": (1, 0, -5)}
    with pytest.raises(OperationDomainValidationError) as ring_error:
        ring_of_integers(malformed)  # type: ignore[arg-type]
    assert ring_error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_field_type"
    )

    with pytest.raises(OperationDomainValidationError) as discriminant_error:
        discriminant(malformed)  # type: ignore[arg-type]
    assert discriminant_error.value.errors()[0]["type"] == (
        "number_field.discriminant_field_type"
    )


class _HugeIndexMatrix:
    def det(self) -> int:
        return 10**300


class _HugeIndexRing:
    matrix = _HugeIndexMatrix()

    def basis_element_pullbacks(self) -> Any:
        raise AssertionError("coordinates expanded despite an oversized index")


def test_integral_basis_index_is_admitted_before_coordinate_expansion() -> None:
    """An index beyond the element envelope is rejected before expansion."""
    recognized = (_HugeIndexRing(), 1, None, 1)
    with pytest.raises(OperationResourceAdmissionError) as error:
        integral_basis_kernel.integral_basis_coordinates(
            cast(Any, None), cast(Any, recognized)
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.integral_basis_coordinate_bound"
    )
