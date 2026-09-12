import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.algebra.trigonometric_rational.operations import (
    TrigonometricRationalSource,
    normalize_trigonometric_rational,
)


def test_pythagorean_identity_normalizes_to_one() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "ADD",
                "children": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "SINE", "angle": {"coefficients": [1]}},
                        "exponent": 2,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "COSINE", "angle": {"coefficients": [1]}},
                        "exponent": 2,
                    },
                ],
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 1
    assert result.numerator.terms[0].exponents == (0,)
    assert (
        result.numerator.terms[0].coefficient == result.denominator.terms[0].coefficient
    )
    assert result.denominator == result.denominator_nonzero


def test_tangent_retains_cosine_nonzero_locus() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [1]}},
                "denominator": {"kind": "COSINE", "angle": {"coefficients": [1]}},
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 2
    assert len(result.denominator.terms) == 2
    assert result.denominator == result.denominator_nonzero


def test_common_laurent_factors_are_cancelled_but_source_locus_is_retained() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [1]}},
                "denominator": {
                    "kind": "SINE",
                    "angle": {"coefficients": [1]},
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.numerator.terms[0].exponents == (0,)
    assert result.denominator.terms[0].exponents == (0,)
    assert len(result.denominator_nonzero.terms) == 2
    assert result.denominator != result.denominator_nonzero


def test_zero_quotient_retains_denominator_nonzero_locus() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 0, "den": 1}},
                "denominator": {"kind": "SINE", "angle": {"coefficients": [1]}},
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert not result.numerator.terms
    assert len(result.denominator.terms) == 1
    assert len(result.denominator_nonzero.terms) == 2


def test_oversized_gaussian_output_is_rejected_by_admission() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": [],
            "expression": {
                "kind": "LITERAL",
                "value": {"num": 10**4096, "den": 1},
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError):
        normalize_trigonometric_rational(request)


def test_oversized_angle_exponent_is_rejected_before_expansion() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "SINE",
                "angle": {"coefficients": [4_097]},
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError):
        normalize_trigonometric_rational(request)


def test_boundary_angle_is_kept_when_no_polynomial_reduction_is_needed() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "SINE",
                "angle": {"coefficients": [4_096]},
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert {term.exponents for term in result.numerator.terms} == {
        (4_096,),
        (-4_096,),
    }


def test_boundary_frequency_cancellation_keeps_the_source_locus() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {
                    "kind": "SINE",
                    "angle": {"coefficients": [4_096]},
                },
                "denominator": {
                    "kind": "SINE",
                    "angle": {"coefficients": [4_096]},
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.denominator.terms[0].exponents == (0,)
    assert {term.exponents for term in result.denominator_nonzero.terms} == {
        (4_096,),
        (-4_096,),
    }


def test_nested_reciprocal_retains_inner_sine_locus() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                "denominator": {
                    "kind": "DIVIDE",
                    "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                    "denominator": {"kind": "SINE", "angle": {"coefficients": [1]}},
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.denominator.terms[0].exponents == (0,)
    assert len(result.denominator_nonzero.terms) == 2
    assert result.denominator != result.denominator_nonzero


def test_zero_power_of_reciprocal_keeps_denominator_locus() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "POWER",
                "base": {
                    "kind": "DIVIDE",
                    "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                    "denominator": {"kind": "SINE", "angle": {"coefficients": [1]}},
                },
                "exponent": 0,
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.numerator.terms[0].exponents == (0,)
    assert result.denominator.terms[0].exponents == (0,)
    assert len(result.denominator_nonzero.terms) == 2
    assert result.denominator != result.denominator_nonzero


def _cosine(coefficient: int) -> dict[str, object]:
    return {"kind": "COSINE", "angle": {"coefficients": [coefficient]}}


def _reciprocal_of_reciprocal(inner: dict[str, object]) -> dict[str, object]:
    one = {"kind": "LITERAL", "value": {"num": 1, "den": 1}}
    return {
        "kind": "DIVIDE",
        "numerator": one,
        "denominator": {
            "kind": "DIVIDE",
            "numerator": one,
            "denominator": inner,
        },
    }


def test_retained_loci_are_admitted_before_the_gcd_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"GCD worker started: {payload}")

    monkeypatch.setattr(
        "jacobian.math.algebra.trigonometric_rational.operations.cancel_common_factor",
        _forbidden,
    )
    product = {
        "kind": "MULTIPLY",
        "children": [
            {
                "kind": "COSINE",
                "angle": {"coefficients": [int(index == axis) for axis in range(7)]},
            }
            for index in range(7)
        ],
    }
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": [f"x{index}" for index in range(7)],
            "expression": {
                "kind": "DIVIDE",
                "numerator": _reciprocal_of_reciprocal(product),
                "denominator": _reciprocal_of_reciprocal(product),
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError):
        normalize_trigonometric_rational(request)


def test_monomial_numerator_skips_the_gcd_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("GCD worker started for a monomial numerator")

    monkeypatch.setattr(
        "jacobian.math.algebra.trigonometric_rational.operations.cancel_common_factor",
        _forbidden,
    )
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                "denominator": {
                    "kind": "MULTIPLY",
                    "children": [_cosine(1), _cosine(2), _cosine(3)],
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 1


def test_reduced_quotient_support_is_bounded_before_gcd() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x", "y"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {
                    "kind": "MULTIPLY",
                    "children": [
                        {"kind": "SINE", "angle": {"coefficients": [65, 0]}},
                        {"kind": "SINE", "angle": {"coefficients": [0, 65]}},
                    ],
                },
                "denominator": {
                    "kind": "MULTIPLY",
                    "children": [
                        {"kind": "SINE", "angle": {"coefficients": [1, 0]}},
                        {"kind": "SINE", "angle": {"coefficients": [0, 1]}},
                    ],
                },
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError, match="support"):
        normalize_trigonometric_rational(request)
