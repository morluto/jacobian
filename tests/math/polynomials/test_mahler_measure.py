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
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials import (
    integer_polynomial_primitive_part,
    mahler_measure,
    quadratic_root_profile,
    reciprocal_profile,
)
from jacobian.math.polynomials._mahler_models import (
    MAX_MAHLER_COEFFICIENT_DIGITS,
    MAX_MAHLER_DEGREE,
    MahlerAlgebraicValue,
    MahlerMeasureRequest,
    MahlerMeasureResult,
    RealQuadraticRootProfileRequest,
    ReciprocalProfileResult,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomial,
    IntegerPolynomialPrimitivePartResult,
)


def _assert_golden_root(value: MahlerAlgebraicValue, index: int) -> None:
    assert isinstance(value, RealAlgebraicValue)
    assert value.polynomial == (1, -1, -1)
    assert value.real_root_index == index


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
    zero = IntegerPolynomial(coefficients=(0,))
    result = integer_polynomial_primitive_part(zero)
    existing = integer_polynomial_primitive_part(zero)
    assert result.content == 0
    assert result.sign == 1
    assert result.degree == 0
    assert result.primitive_part.coefficients == (0,)
    assert result.reconstruction.coefficients == (0,)
    assert result.content == existing.content
    assert result.primitive_part == existing.primitive_part
    assert result.reconstruction == existing.reconstruction
    restored = IntegerPolynomialPrimitivePartResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_mahler_coefficient_digit_bound_is_an_admission_error() -> None:
    oversized = 10**MAX_MAHLER_COEFFICIENT_DIGITS
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        mahler_measure(IntegerPolynomial(coefficients=(oversized, -1, -1)))
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, oversized)))


def test_mahler_result_does_not_replay_coefficient_digit_admission() -> None:
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1, -1)))
    forged = result.model_dump(mode="json")
    forged["polynomial"]["coefficients"] = [
        str(10**MAX_MAHLER_COEFFICIENT_DIGITS),
        "-1",
        "-1",
    ]
    forged["leading_coefficient"] = str(10**MAX_MAHLER_COEFFICIENT_DIGITS)
    restored = MahlerMeasureResult.model_validate_json(
        encode_strict_json(forged), strict=True
    )
    assert restored.polynomial.coefficients[0] == 10**MAX_MAHLER_COEFFICIENT_DIGITS


def test_reciprocal_profile_accepts_negative_leading_source() -> None:
    result = reciprocal_profile(IntegerPolynomial(coefficients=(-1, 0, -1)))
    assert result.state == "RECIPROCAL"


def test_reciprocal_profile_identifies_palindromic_and_antipalindromic() -> None:
    """Reciprocal and antireciprocal states are distinguished exactly."""
    palindromic = reciprocal_profile(IntegerPolynomial(coefficients=(1, 2, 1)))
    assert palindromic.state == "RECIPROCAL"
    assert palindromic.reversed_coefficients == (1, 2, 1)
    assert palindromic.coefficient_pair_ledger == ((1, 1), (2, 2))

    antipalindromic = reciprocal_profile(IntegerPolynomial(coefficients=(1, 0, -1)))
    assert antipalindromic.state == "ANTIRECIPROCAL"
    assert antipalindromic.reversed_coefficients == (-1, 0, 1)


def test_reciprocal_profile_reports_neither_when_no_structure_holds() -> None:
    """A generic polynomial has neither reciprocal state."""
    result = reciprocal_profile(IntegerPolynomial(coefficients=(1, 3, 2)))
    assert result.state == "NEITHER"
    assert result.leading_coefficient == 1
    assert result.constant_coefficient == 2


def test_quadratic_root_profile_classifies_each_real_root() -> None:
    """x^2-x-1 has one root inside and one outside the closed unit disk."""
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(1, -1, -1)))
    assert result.discriminant == 5
    assert result.root_kind == "DISTINCT_REAL"
    assert result.sum_of_roots == CanonicalRational(num=1, den=1)
    assert result.product_of_roots == CanonicalRational(num=-1, den=1)
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    _assert_golden_root(result.roots[0], 0)
    _assert_golden_root(result.roots[1], 1)


def test_quadratic_root_profile_reports_on_circle_roots() -> None:
    """x^2+1 has a conjugate pair with squared modulus exactly one."""
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, 1)))
    assert result.root_kind == "COMPLEX_CONJUGATE"
    assert result.root_locations == ("ON_UNIT_CIRCLE",)


def test_mahler_measure_of_golden_quadratic_is_the_golden_ratio() -> None:
    """M(x^2-x-1) = (1+sqrt(5))/2."""
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1, -1)))
    _assert_golden_root(result.mahler_measure, 1)
    assert result.leading_coefficient == 1


def test_mahler_measure_keeps_the_leading_coefficient() -> None:
    """M(2x^2-2x-2) = 2*phi = 1+sqrt(5), not the monic value."""
    result = mahler_measure(IntegerPolynomial(coefficients=(2, -2, -2)))
    assert isinstance(result.mahler_measure, RealAlgebraicValue)
    assert result.mahler_measure.polynomial == (1, -2, -4)
    assert result.mahler_measure.real_root_index == 1
    assert result.leading_coefficient == 2
    # The audited mistake (dropping |a_d|) would return the monic measure.
    assert (
        result.mahler_measure
        != mahler_measure(IntegerPolynomial(coefficients=(1, -1, -1))).mahler_measure
    )


def test_mahler_measure_of_on_circle_polynomial_is_one() -> None:
    """M(x^2+x+1) = 1 because every root lies on the unit circle."""
    result = mahler_measure(IntegerPolynomial(coefficients=(1, 1, 1)))
    assert result.mahler_measure == CanonicalRational(num=1, den=1)


def test_mahler_measure_accepts_a_pure_quadratic_monomial() -> None:
    """Trailing zero coefficients do not make a nonzero polynomial zero."""
    result = mahler_measure(IntegerPolynomial(coefficients=(3, 0, 0)))
    assert result.mahler_measure == CanonicalRational(num=3, den=1)
    assert result.root_locations == ("INSIDE_UNIT_DISK", "INSIDE_UNIT_DISK")


def test_mahler_measure_uses_the_absolute_leading_coefficient() -> None:
    """M(3x^2-3) = 3 * max(1,1)^2 = 3."""
    result = mahler_measure(IntegerPolynomial(coefficients=(3, 0, -3)))
    assert result.mahler_measure == CanonicalRational(num=3, den=1)


def test_mahler_measure_of_a_linear_polynomial() -> None:
    """M(2x-4) = 2 * |2| = 4 for the single outside root two."""
    result = mahler_measure(IntegerPolynomial(coefficients=(2, -4)))
    assert result.mahler_measure == CanonicalRational(num=4, den=1)
    assert result.root_locations == ("OUTSIDE_UNIT_DISK",)


def test_mahler_measure_of_a_nonzero_constant_is_its_absolute_value() -> None:
    result = mahler_measure(IntegerPolynomial(coefficients=(-5,)))
    assert result.degree == 0
    assert result.root_locations == ()
    assert result.mahler_measure == CanonicalRational(num=5, den=1)


def test_mahler_measure_of_a_unit_root_linear_polynomial() -> None:
    """M(x-1) = 1 because the only root lies on the unit circle."""
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1)))
    assert result.mahler_measure == CanonicalRational(num=1, den=1)
    assert result.root_locations == ("ON_UNIT_CIRCLE",)


def test_quadratic_roots_use_the_canonical_algebraic_carrier() -> None:
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, -20)))
    first, second = result.roots
    assert isinstance(first, RealAlgebraicValue)
    assert isinstance(second, RealAlgebraicValue)
    assert first.polynomial == (1, 0, -20)
    assert first.real_root_index == 0
    assert second.real_root_index == 1


def test_negative_leading_quadratic_roots_are_increasing() -> None:
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(-1, 1, 1)))
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    _assert_golden_root(result.roots[0], 0)
    _assert_golden_root(result.roots[1], 1)


def test_result_round_trips_through_strict_json() -> None:
    """The exact measure survives strict JSON serialization unchanged."""
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1, -1)))
    restored = MahlerMeasureResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
    content = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    assert (
        IntegerPolynomialPrimitivePartResult.model_validate_json(
            encode_strict_json(content.model_dump(mode="json")), strict=True
        )
        == content
    )
    reciprocal = reciprocal_profile(IntegerPolynomial(coefficients=(1, 0, 1)))
    assert (
        ReciprocalProfileResult.model_validate_json(
            encode_strict_json(reciprocal.model_dump(mode="json")), strict=True
        )
        == reciprocal
    )


def test_zero_quadratic_leading_coefficient_is_rejected() -> None:
    """A quadratic profile needs a nonzero leading coefficient."""
    with pytest.raises(ValidationError):
        RealQuadraticRootProfileRequest(
            polynomial=IntegerPolynomial(coefficients=(0, 1, 1))
        )
    with pytest.raises(OperationDomainValidationError):
        quadratic_root_profile(
            IntegerPolynomial.model_construct(coefficients=(0, 1, 1))
        )


def test_reciprocal_result_rejects_inconsistent_pair_ledger() -> None:
    with pytest.raises(ValidationError):
        ReciprocalProfileResult(
            degree=3,
            reversed_coefficients=(1, 2, 3, 1),
            state="NEITHER",
            leading_coefficient=1,
            constant_coefficient=1,
            coefficient_pair_ledger=((1, 1), (2, 3)),
        )


def test_reciprocal_result_does_not_replay_classification() -> None:
    result = reciprocal_profile(IntegerPolynomial(coefficients=(1, 0, 1)))
    forged = result.model_dump(mode="json")
    forged["state"] = "NEITHER"
    restored = ReciprocalProfileResult.model_validate_json(
        encode_strict_json(forged), strict=True
    )
    assert restored.state == "NEITHER"


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
    forged["primitive_part"]["coefficients"] = ["2", "0", "-2"]
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


def test_linear_profiles_admit_carrier_length_beyond_mahler_degree() -> None:
    coefficients = (1,) + (0,) * MAX_MAHLER_DEGREE + (1,)
    polynomial = IntegerPolynomial(coefficients=coefficients)
    assert len(polynomial.coefficients) == MAX_MAHLER_DEGREE + 2
    content = integer_polynomial_primitive_part(polynomial)
    assert content.degree == MAX_MAHLER_DEGREE + 1
    assert content.reconstruction == polynomial
    reciprocal = reciprocal_profile(polynomial)
    assert reciprocal.degree == MAX_MAHLER_DEGREE + 1
    assert reciprocal.state == "RECIPROCAL"
    with pytest.raises(ValidationError, match="degree at most"):
        MahlerMeasureRequest(polynomial=polynomial)
    with pytest.raises(OperationDomainValidationError, match="degree at most"):
        mahler_measure(polynomial)


def test_content_profile_admits_carrier_length_beyond_elementary_degree() -> None:
    polynomial = IntegerPolynomial(coefficients=(1,) + (0,) * 128)
    assert len(polynomial.coefficients) == 129
    result = integer_polynomial_primitive_part(polynomial)
    assert result.degree == 128
    assert result.reconstruction == polynomial


def test_mahler_measure_rejects_empty_native_coefficients() -> None:
    forged = IntegerPolynomial.model_construct(coefficients=())
    with pytest.raises(
        OperationDomainValidationError, match="at least one coefficient"
    ):
        mahler_measure(forged)
    with pytest.raises(
        OperationDomainValidationError, match="at least one coefficient"
    ):
        reciprocal_profile(forged)


def test_mahler_measure_rejects_leading_zero_native_coefficients() -> None:
    forged = IntegerPolynomial.model_construct(coefficients=(0, 1))
    with pytest.raises(OperationDomainValidationError, match="leading zeros"):
        mahler_measure(forged)
    with pytest.raises(OperationDomainValidationError, match="leading zeros"):
        integer_polynomial_primitive_part(forged)


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


def test_reciprocal_profile_preflights_duplicated_coefficient_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.polynomials._mahler_kernel.MAX_PROFILE_RESULT_DIGITS",
        20,
    )
    with pytest.raises(OperationResourceAdmissionError, match="output bound"):
        reciprocal_profile(IntegerPolynomial(coefficients=(10**12, 10**12 + 1)))


def test_reciprocal_profile_is_not_a_public_catalog_operation() -> None:
    from jacobian.math.polynomials._tools import TOOLS

    assert all(
        operation.operation_id != "polynomial.reciprocal_profile.compute"
        for operation in TOOLS
    )


def test_mahler_result_binds_degree_leading_coefficient_and_locations() -> None:
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1)))

    forged = result.model_dump(mode="json")
    forged["degree"] = 2
    with pytest.raises(ValidationError):
        MahlerMeasureResult.model_validate_json(encode_strict_json(forged), strict=True)


def test_mahler_result_validator_does_not_replay_measure_mathematics() -> None:
    """The trusted kernel owns the value; the model checks only source shape."""
    result = mahler_measure(IntegerPolynomial(coefficients=(1, -1)))
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


def test_large_nonsquare_discriminant_retains_normalized_radicand() -> None:
    polynomial = IntegerPolynomial(coefficients=(1, -(10**100), -1))
    profile = quadratic_root_profile(polynomial)
    assert profile.discriminant == 10**200 + 4
    assert profile.root_kind == "DISTINCT_REAL"
    assert profile.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    assert all(
        isinstance(root, RealAlgebraicValue) and root.polynomial == (1, -(10**100), -1)
        for root in profile.roots
    )
    measure = mahler_measure(polynomial)
    assert isinstance(measure.mahler_measure, RealAlgebraicValue)
    assert measure.mahler_measure == profile.roots[1]


def test_large_nonsquare_product_is_exact() -> None:
    result = mahler_measure(IntegerPolynomial(coefficients=(1, 1, -(10**24))))
    assert result.mahler_measure == CanonicalRational(num=10**24, den=1)


def test_complex_pair_has_modulus_instead_of_fake_real_root() -> None:
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, 4)))
    assert result.roots == ()
    assert result.complex_pair_squared_modulus is not None
    assert result.complex_pair_squared_modulus.as_fraction() == 4
    assert mahler_measure(
        IntegerPolynomial(coefficients=(1, 0, 4))
    ).mahler_measure == CanonicalRational(num=4, den=1)


def test_scaled_quadratic_normalizes_content_before_surd_admission() -> None:
    scale = 10**20
    result = quadratic_root_profile(
        IntegerPolynomial(coefficients=(scale, -scale, -scale))
    )
    assert result.discriminant == 5 * scale * scale
    assert result.root_locations == ("INSIDE_UNIT_DISK", "OUTSIDE_UNIT_DISK")
    assert all(
        isinstance(root, RealAlgebraicValue) and root.polynomial == (1, -1, -1)
        for root in result.roots
    )


def test_perfect_square_discriminant_keeps_its_rational_roots() -> None:
    result = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, -9)))
    assert result.roots == (
        CanonicalRational(num=-3, den=1),
        CanonicalRational(num=3, den=1),
    )


def test_large_perfect_square_discriminant_avoids_surd_admission() -> None:
    root = 10**20
    result = quadratic_root_profile(
        IntegerPolynomial(coefficients=(1, 0, -(root * root)))
    )
    assert result.roots == (
        CanonicalRational(num=-root, den=1),
        CanonicalRational(num=root, den=1),
    )


def test_native_mahler_family_rejects_non_polynomial_arguments() -> None:
    with pytest.raises(OperationDomainValidationError, match="canonical integer"):
        mahler_measure((1, -1, -1))  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError, match="canonical integer"):
        quadratic_root_profile((1, -1, -1))  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError, match="canonical integer"):
        reciprocal_profile((1, 0, 1))  # type: ignore[arg-type]


def test_catalog_mahler_family_matches_domain_valued_natives() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    polynomial = IntegerPolynomial(coefficients=(1, -1, -1))
    catalog = Catalog.open()
    native_measure = mahler_measure(polynomial)
    dispatched_measure = invoke_operation(
        "polynomial.mahler_measure.compute",
        {"polynomial": {"coefficients": ["1", "-1", "-1"]}},
        catalog,
    )
    assert dispatched_measure.output == native_measure.model_dump(mode="json")
    native_roots = quadratic_root_profile(polynomial)
    dispatched_roots = invoke_operation(
        "polynomial.quadratic.real_root_profile.compute",
        {"polynomial": {"coefficients": ["1", "-1", "-1"]}},
        catalog,
    )
    assert dispatched_roots.output == native_roots.model_dump(mode="json")
    native_content = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    dispatched_content = invoke_operation(
        "polynomial.integer.content_primitive_profile.compute",
        {"polynomial": {"coefficients": ["6", "0", "-6"]}},
        catalog,
    )
    assert dispatched_content.output == native_content.model_dump(mode="json")
    assert dispatched_content.output["convention"] == (
        "NONNEGATIVE_CONTENT_POSITIVE_LEADING"
    )
