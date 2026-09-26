"""Exact integer-polynomial content and primitive-part contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import integer_polynomial_primitive_part
from jacobian.math.polynomials._models import (
    IntegerPolynomial,
    IntegerPolynomialPrimitivePartResult,
)
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS


def test_integer_polynomial_primitive_part_reconstructs_the_source() -> None:
    """sign * content * primitive_part equals the input polynomial exactly."""
    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6, 12))
    )
    assert result.sign == 1
    assert result.content == 6
    assert result.primitive_part.coefficients == (1, 0, -1, 2)
    assert result.reconstruction.coefficients == (6, 0, -6, 12)
    assert result.degree == 3


def test_returned_primitive_part_composes_with_polynomial_operations() -> None:
    """Profile outputs reuse the canonical integer-polynomial carrier."""
    from jacobian.math.polynomials import integer_polynomial_content

    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    assert integer_polynomial_content(result.primitive_part).content == 1
    assert integer_polynomial_content(result.reconstruction).content == 6


def test_integer_polynomial_primitive_part_is_positive_leading() -> None:
    """The primitive part requires a positive leading coefficient."""
    with pytest.raises(ValidationError):
        IntegerPolynomialPrimitivePartResult(
            sign=1,
            content=6,
            primitive_part=IntegerPolynomial(coefficients=(-1, 0, 1)),
            degree=2,
            reconstruction=IntegerPolynomial(coefficients=(-6, 0, 6)),
        )
    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    assert result.sign == 1
    assert result.primitive_part.coefficients[0] > 0


def test_content_profile_retains_negative_source_sign() -> None:
    """Content extraction must accept and reconstruct a negative-leading source."""
    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(-6, 0, 6))
    )
    assert result.sign == -1
    assert result.content == 6
    assert result.primitive_part.coefficients == (1, 0, -1)
    assert result.reconstruction.coefficients == (-6, 0, 6)


def test_content_profile_matches_primitive_part_on_the_zero_polynomial() -> None:
    result = integer_polynomial_primitive_part(IntegerPolynomial(coefficients=(0,)))
    assert result.content == 0
    assert result.sign == 1
    assert result.degree == 0
    assert result.primitive_part.coefficients == (0,)
    assert result.reconstruction.coefficients == (0,)
    restored = IntegerPolynomialPrimitivePartResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_content_profile_rejects_a_non_carrier_native_argument() -> None:
    """A native boundary classifies the wrong type as a domain error."""
    with pytest.raises(OperationDomainValidationError) as exc_info:
        integer_polynomial_primitive_part((1, 2))  # type: ignore[arg-type]
    assert exc_info.value.errors()[0]["type"] == "polynomial.primitive_part_carrier"


def test_content_profile_reports_the_duplicated_output_envelope() -> None:
    """A carrier-valid polynomial can still exceed the retained digit bound."""
    coefficients = tuple(10**1000 + index for index in range(MAX_POLYNOMIAL_TERMS))
    polynomial = IntegerPolynomial(coefficients=coefficients)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        integer_polynomial_primitive_part(polynomial)
    assert exc_info.value.errors()[0]["type"] == (
        "polynomial.content_profile_result_digits"
    )


def test_content_result_rejects_negative_or_inconsistent_reconstruction() -> None:
    source = IntegerPolynomial(coefficients=(6, 0, -6))
    result = integer_polynomial_primitive_part(source)
    forged = result.model_dump(mode="json")
    forged["content"] = -6
    with pytest.raises(ValidationError):
        IntegerPolynomialPrimitivePartResult.model_validate_json(
            encode_strict_json(forged), strict=True
        )

    forged = result.model_dump(mode="json")
    forged["primitive_part"]["coefficients"] = ["2", "0", "-2", "1"]
    with pytest.raises(ValidationError):
        IntegerPolynomialPrimitivePartResult.model_validate_json(
            encode_strict_json(forged), strict=True
        )


def test_content_result_does_not_replay_primitivity() -> None:
    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    forged = result.model_dump(mode="json")
    forged["content"] = "3"
    forged["primitive_part"]["coefficients"] = ["2", "0", "-2"]
    restored = IntegerPolynomialPrimitivePartResult.model_validate_json(
        encode_strict_json(forged), strict=True
    )
    assert restored.content == 3
    assert restored.primitive_part.coefficients == (2, 0, -2)
    forged = result.model_dump(mode="json")
    forged["primitive_part"]["coefficients"] = ["1", "0", "-1"]
    restored = IntegerPolynomialPrimitivePartResult.model_validate_json(
        encode_strict_json(forged), strict=True
    )
    assert restored.primitive_part.coefficients == (1, 0, -1)
    assert restored.reconstruction.coefficients == (6, 0, -6)


def test_content_profile_admits_carrier_length_beyond_elementary_degree() -> None:
    polynomial = IntegerPolynomial(coefficients=(1,) + (0,) * 128)
    assert len(polynomial.coefficients) == 129
    result = integer_polynomial_primitive_part(polynomial)
    assert result.degree == 128
    assert result.reconstruction == polynomial


def test_content_profile_preflights_duplicated_coefficient_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.polynomials._elementary_kernel.MAX_PRIMITIVE_PART_RESULT_DIGITS",
        20,
    )
    with pytest.raises(OperationResourceAdmissionError, match="output bound"):
        integer_polynomial_primitive_part(
            IntegerPolynomial(coefficients=(10**12, 10**12 + 1))
        )


def test_content_accepts_carrier_length_beyond_mahler_degree() -> None:
    coefficients = (1,) + (0,) * 128 + (1,)
    polynomial = IntegerPolynomial(coefficients=coefficients)
    result = integer_polynomial_primitive_part(polynomial)
    assert result.degree == 129
    assert result.reconstruction == polynomial


def test_content_rejects_forged_carriers_beyond_integer_envelope() -> None:
    oversized = IntegerPolynomial.model_construct(
        coefficients=(1,) * (MAX_POLYNOMIAL_TERMS + 1)
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="integer-polynomial carrier"
    ):
        integer_polynomial_primitive_part(oversized)
    too_wide = IntegerPolynomial.model_construct(
        coefficients=(10**MAX_CANONICAL_INTEGER_DIGITS,)
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="canonical integer representation"
    ):
        integer_polynomial_primitive_part(too_wide)


def test_content_profile_rejects_leading_zero_native_coefficients() -> None:
    forged = IntegerPolynomial.model_construct(coefficients=(0, 1))
    with pytest.raises(OperationDomainValidationError, match="leading zeros"):
        integer_polynomial_primitive_part(forged)


def test_nonzero_content_result_round_trips_through_strict_json() -> None:
    result = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    restored = IntegerPolynomialPrimitivePartResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
