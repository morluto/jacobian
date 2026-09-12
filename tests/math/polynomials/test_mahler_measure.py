"""Exact polynomial content, reciprocal, root-location, and Mahler profiles (#1787)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials import _mahler_kernel
from jacobian.math.polynomials._mahler_kernel import (
    content_primitive_profile,
    mahler_measure,
    quadratic_root_profile,
    reciprocal_profile,
)
from jacobian.math.polynomials._mahler_models import (
    MAX_MAHLER_RADICAND_BITS,
    ContentPrimitiveProfileRequest,
    ContentPrimitiveProfileResult,
    IntegerPolynomialProfileValue,
    IntegerPolynomialValue,
    MahlerAlgebraicValue,
    MahlerMeasureRequest,
    MahlerMeasureResult,
    RealQuadraticRootProfileRequest,
    ReciprocalProfileRequest,
    ReciprocalProfileResult,
)


def _assert_golden_root(value: MahlerAlgebraicValue, index: int) -> None:
    assert isinstance(value, RealAlgebraicValue)
    assert value.polynomial == (1, -1, -1)
    assert value.real_root_index == index


def test_content_primitive_profile_reconstructs_the_source() -> None:
    """sign * content * primitive_part equals the input polynomial exactly."""
    result = content_primitive_profile(
        ContentPrimitiveProfileRequest(
            polynomial=IntegerPolynomialProfileValue(
                coefficients_descending=(6, 0, -6, 12)
            )
        )
    )
    assert result.sign == 1
    assert result.content == 6
    assert result.primitive_part.coefficients_descending == (1, 0, -1, 2)
    assert result.reconstruction.coefficients_descending == (6, 0, -6, 12)
    assert result.degree == 3


def test_content_primitive_profile_is_positive_leading() -> None:
    """The profile carrier requires and returns a positive leading coefficient."""
    with pytest.raises(ValidationError):
        IntegerPolynomialProfileValue(coefficients_descending=(-6, 0, 6))
    result = content_primitive_profile(
        ContentPrimitiveProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(6, 0, -6))
        )
    )
    assert result.sign == 1
    assert result.primitive_part.coefficients_descending[0] > 0


def test_content_profile_retains_negative_source_sign() -> None:
    """Content extraction must accept and reconstruct a negative-leading source."""
    result = content_primitive_profile(
        ContentPrimitiveProfileRequest(
            polynomial=IntegerPolynomialValue(coefficients_descending=(-6, 0, 6))
        )
    )
    assert result.sign == -1
    assert result.content == 6
    assert result.primitive_part.coefficients_descending == (1, 0, -1)
    assert result.reconstruction.coefficients_descending == (-6, 0, 6)


def test_reciprocal_profile_accepts_negative_leading_source() -> None:
    result = reciprocal_profile(
        ReciprocalProfileRequest(
            polynomial=IntegerPolynomialValue(coefficients_descending=(-1, 0, -1))
        )
    )
    assert result.state == "RECIPROCAL"


def test_reciprocal_profile_identifies_palindromic_and_antipalindromic() -> None:
    """Reciprocal and antireciprocal states are distinguished exactly."""
    palindromic = reciprocal_profile(
        ReciprocalProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(1, 2, 1))
        )
    )
    assert palindromic.state == "RECIPROCAL"
    assert palindromic.reversed_coefficients == (1, 2, 1)
    assert palindromic.coefficient_pair_ledger == ((1, 1), (2, 2))

    antipalindromic = reciprocal_profile(
        ReciprocalProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(1, 0, -1))
        )
    )
    assert antipalindromic.state == "ANTIRECIPROCAL"
    assert antipalindromic.reversed_coefficients == (-1, 0, 1)


def test_reciprocal_profile_reports_neither_when_no_structure_holds() -> None:
    """A generic polynomial has neither reciprocal state."""
    result = reciprocal_profile(
        ReciprocalProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(1, 3, 2))
        )
    )
    assert result.state == "NEITHER"
    assert result.leading_coefficient == 1
    assert result.constant_coefficient == 2


def test_quadratic_root_profile_classifies_each_real_root() -> None:
    """x^2-x-1 has one root inside and one outside the closed unit disk."""
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, -1, -1))
    )
    assert result.discriminant == 5
    assert result.root_kind == "DISTINCT_REAL"
    assert result.sum_of_roots == CanonicalRational(num=1, den=1)
    assert result.product_of_roots == CanonicalRational(num=-1, den=1)
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    _assert_golden_root(result.roots[0], 0)
    _assert_golden_root(result.roots[1], 1)


def test_quadratic_root_profile_reports_on_circle_roots() -> None:
    """x^2+1 has a conjugate pair with squared modulus exactly one."""
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, 1))
    )
    assert result.root_kind == "COMPLEX_CONJUGATE"
    assert result.root_locations == ("ON_UNIT_CIRCLE",)


def test_mahler_measure_of_golden_quadratic_is_the_golden_ratio() -> None:
    """M(x^2-x-1) = (1+sqrt(5))/2."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, -1, -1)))
    _assert_golden_root(result.mahler_measure, 1)
    assert result.leading_coefficient == 1


def test_mahler_measure_keeps_the_leading_coefficient() -> None:
    """M(2x^2-2x-2) = 2*phi = 1+sqrt(5), not the monic value."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(2, -2, -2)))
    assert isinstance(result.mahler_measure, RealAlgebraicValue)
    assert result.mahler_measure.polynomial == (1, -2, -4)
    assert result.mahler_measure.real_root_index == 1
    assert result.leading_coefficient == 2
    # The audited mistake (dropping |a_d|) would return the monic measure.
    assert (
        result.mahler_measure
        != mahler_measure(
            MahlerMeasureRequest(coefficients_descending=(1, -1, -1))
        ).mahler_measure
    )


def test_mahler_measure_of_on_circle_polynomial_is_one() -> None:
    """M(x^2+x+1) = 1 because every root lies on the unit circle."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, 1, 1)))
    assert result.mahler_measure == CanonicalRational(num=1, den=1)


def test_mahler_measure_accepts_a_pure_quadratic_monomial() -> None:
    """Trailing zero coefficients do not make a nonzero polynomial zero."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(3, 0, 0)))
    assert result.mahler_measure == CanonicalRational(num=3, den=1)
    assert result.root_locations == ("INSIDE_UNIT_DISK", "INSIDE_UNIT_DISK")


def test_mahler_measure_uses_the_absolute_leading_coefficient() -> None:
    """M(3x^2-3) = 3 * max(1,1)^2 = 3."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(3, 0, -3)))
    assert result.mahler_measure == CanonicalRational(num=3, den=1)


def test_mahler_measure_of_a_linear_polynomial() -> None:
    """M(2x-4) = 2 * |2| = 4 for the single outside root two."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(2, -4)))
    assert result.mahler_measure == CanonicalRational(num=4, den=1)
    assert result.root_locations == ("OUTSIDE_UNIT_DISK",)


def test_mahler_measure_of_a_nonzero_constant_is_its_absolute_value() -> None:
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(-5,)))
    assert result.degree == 0
    assert result.root_locations == ()
    assert result.mahler_measure == CanonicalRational(num=5, den=1)


def test_mahler_measure_of_a_unit_root_linear_polynomial() -> None:
    """M(x-1) = 1 because the only root lies on the unit circle."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, -1)))
    assert result.mahler_measure == CanonicalRational(num=1, den=1)
    assert result.root_locations == ("ON_UNIT_CIRCLE",)


def test_quadratic_roots_use_the_canonical_algebraic_carrier() -> None:
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, -20))
    )
    first, second = result.roots
    assert isinstance(first, RealAlgebraicValue)
    assert isinstance(second, RealAlgebraicValue)
    assert first.polynomial == (1, 0, -20)
    assert first.real_root_index == 0
    assert second.real_root_index == 1


def test_negative_leading_quadratic_roots_are_increasing() -> None:
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(-1, 1, 1))
    )
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    _assert_golden_root(result.roots[0], 0)
    _assert_golden_root(result.roots[1], 1)


def test_result_round_trips_through_strict_json() -> None:
    """The exact measure survives strict JSON serialization unchanged."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, -1, -1)))
    restored = MahlerMeasureResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
    content = content_primitive_profile(
        ContentPrimitiveProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(6, 0, -6))
        )
    )
    assert (
        ContentPrimitiveProfileResult.model_validate_json(
            encode_strict_json(content.model_dump(mode="json")), strict=True
        )
        == content
    )
    reciprocal = reciprocal_profile(
        ReciprocalProfileRequest(
            polynomial=IntegerPolynomialProfileValue(coefficients_descending=(1, 0, 1))
        )
    )
    assert (
        ReciprocalProfileResult.model_validate_json(
            encode_strict_json(reciprocal.model_dump(mode="json")), strict=True
        )
        == reciprocal
    )


def test_zero_quadratic_leading_coefficient_is_rejected() -> None:
    """A quadratic profile needs a nonzero leading coefficient."""
    with pytest.raises(ValidationError):
        RealQuadraticRootProfileRequest(coefficients_descending=(0, 1, 1))
    with pytest.raises(OperationDomainValidationError):
        quadratic_root_profile(
            RealQuadraticRootProfileRequest.model_construct(
                coefficients_descending=(0, 1, 1)
            )
        )


def test_reciprocal_result_rejects_endpoint_only_forgery() -> None:
    """Equal endpoints alone cannot establish reciprocal structure."""
    with pytest.raises(ValidationError):
        ReciprocalProfileResult(
            degree=3,
            reversed_coefficients=(1, 2, 3, 1),
            state="RECIPROCAL",
            leading_coefficient=1,
            constant_coefficient=1,
            coefficient_pair_ledger=((1, 1), (3, 2)),
        )


def test_content_result_rejects_negative_or_nonprimitive_claims() -> None:
    source = IntegerPolynomialValue(coefficients_descending=(6, 0, -6))
    result = content_primitive_profile(
        ContentPrimitiveProfileRequest(polynomial=source)
    )
    forged = result.model_dump(mode="json")
    forged["content"] = -6
    with pytest.raises(ValidationError):
        ContentPrimitiveProfileResult.model_validate_json(
            encode_strict_json(forged), strict=True
        )

    forged = result.model_dump(mode="json")
    forged["primitive_part"]["coefficients_descending"] = ["2", "0", "-2"]
    with pytest.raises(ValidationError):
        ContentPrimitiveProfileResult.model_validate_json(
            encode_strict_json(forged), strict=True
        )


def test_mahler_result_binds_degree_leading_coefficient_and_locations() -> None:
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, -1)))

    forged = result.model_dump(mode="json")
    forged["degree"] = 2
    with pytest.raises(ValidationError):
        MahlerMeasureResult.model_validate_json(encode_strict_json(forged), strict=True)


def test_mahler_result_validator_does_not_replay_measure_mathematics() -> None:
    """The trusted kernel owns the value; the model checks only source shape."""
    result = mahler_measure(MahlerMeasureRequest(coefficients_descending=(1, -1)))
    forged = result.model_dump(mode="json")
    forged["mahler_measure"] = {"num": "99", "den": "1"}
    restored = MahlerMeasureResult.model_validate_json(
        encode_strict_json(forged), strict=True
    )
    assert restored.mahler_measure == CanonicalRational(num=99, den=1)

    forged = result.model_dump(mode="json")
    forged["leading_coefficient"] = 99
    with pytest.raises(ValidationError):
        MahlerMeasureResult.model_validate_json(encode_strict_json(forged), strict=True)

    forged = result.model_dump(mode="json")
    forged["root_locations"] = ["UNRESOLVED"]
    with pytest.raises(ValidationError):
        MahlerMeasureResult.model_validate_json(encode_strict_json(forged), strict=True)


def test_quadratic_discriminant_is_admitted_before_isqrt() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        quadratic_root_profile(
            RealQuadraticRootProfileRequest(
                coefficients_descending=(
                    1,
                    0,
                    -((1 << MAX_MAHLER_RADICAND_BITS) + 1),
                )
            )
        )


def test_complex_pair_has_modulus_instead_of_fake_real_root() -> None:
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, 4))
    )
    assert result.roots == ()
    assert result.complex_pair_squared_modulus is not None
    assert result.complex_pair_squared_modulus.as_fraction() == 4
    assert mahler_measure(
        MahlerMeasureRequest(coefficients_descending=(1, 0, 4))
    ).mahler_measure == CanonicalRational(num=4, den=1)


def test_large_nonsquare_discriminant_uses_bounded_factorization() -> None:
    result = mahler_measure(
        MahlerMeasureRequest(coefficients_descending=(1, 1, -(10**24)))
    )
    assert result.mahler_measure == CanonicalRational(num=10**24, den=1)


def test_scaled_quadratic_normalizes_content_before_surd_admission() -> None:
    scale = 10**20
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(scale, -scale, -scale))
    )
    assert result.discriminant == 5 * scale * scale
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    assert all(
        isinstance(root, RealAlgebraicValue) and root.polynomial == (1, -1, -1)
        for root in result.roots
    )


def test_quadratic_surd_normalization_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = _mahler_kernel.factorize_certified

    def count_calls(request: CertifiedFactorizationRequest) -> object:
        calls.append(request.value)
        return original(request)

    monkeypatch.setattr(
        "jacobian.math.polynomials._mahler_kernel.factorize_certified", count_calls
    )
    quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, -1000003))
    )
    assert calls == [4 * 1000003]


def test_factorization_backend_failure_is_a_typed_resource_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_request: object) -> None:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(
        "jacobian.math.polynomials._mahler_kernel.factorize_certified", fail
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        quadratic_root_profile(
            RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, -1000003))
        )
    assert error.value.errors()[0]["type"] == "polynomial.mahler_factorization_backend"


def test_perfect_square_discriminant_keeps_its_rational_roots() -> None:
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, -9))
    )
    assert result.roots == (
        CanonicalRational(num=-3, den=1),
        CanonicalRational(num=3, den=1),
    )


def test_large_perfect_square_discriminant_avoids_surd_admission() -> None:
    root = 10**20
    result = quadratic_root_profile(
        RealQuadraticRootProfileRequest(coefficients_descending=(1, 0, -(root * root)))
    )
    assert result.roots == (
        CanonicalRational(num=-root, den=1),
        CanonicalRational(num=root, den=1),
    )
