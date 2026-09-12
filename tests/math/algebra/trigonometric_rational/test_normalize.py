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
    def _product(first_frequency: int) -> dict[str, object]:
        return {
            "kind": "MULTIPLY",
            "children": [
                {
                    "kind": "COSINE",
                    "angle": {
                        "coefficients": [
                            first_frequency if axis == 0 else int(index == axis)
                            for axis in range(7)
                        ]
                    },
                }
                for index in range(7)
            ],
        }

    request = TrigonometricRationalSource.model_validate(
        {
            "variables": [f"x{index}" for index in range(7)],
            "expression": {
                "kind": "DIVIDE",
                "numerator": _reciprocal_of_reciprocal(_product(1)),
                "denominator": _reciprocal_of_reciprocal(_product(2)),
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


def test_univariate_lattice_stride_admits_sin_4096_over_sin() -> None:
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [4096]}},
                "denominator": {"kind": "SINE", "angle": {"coefficients": [1]}},
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 4096
    assert len(result.denominator.terms) == 1


def _scaled_sine(scale: int, coefficient: int = 1) -> dict[str, object]:
    return {
        "kind": "MULTIPLY",
        "children": [
            {"kind": "LITERAL", "value": {"num": scale, "den": 1}},
            {"kind": "SINE", "angle": {"coefficients": [coefficient]}},
        ],
    }


def test_proportional_large_scalars_cancel_without_admitted_cross_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("GCD worker started for proportional Laurent operands")

    monkeypatch.setattr(
        "jacobian.math.algebra.trigonometric_rational.operations.cancel_common_factor",
        _forbidden,
    )
    scaled = _scaled_sine(10**3000)
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": scaled,
                "denominator": scaled,
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.numerator.terms[0].exponents == (0,)
    assert result.denominator.terms[0].exponents == (0,)
    assert (
        result.numerator.terms[0].coefficient == result.denominator.terms[0].coefficient
    )
    assert {term.exponents for term in result.denominator_nonzero.terms} == {
        (0,),
        (2,),
    }
    assert all(
        abs(term.coefficient.real.num) < 10
        and abs(term.coefficient.imaginary.num) < 10
        for term in result.denominator_nonzero.terms
    )


def test_locus_scalar_units_are_normalized_before_combining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("GCD worker started after scalar-unit locus cancellation")

    monkeypatch.setattr(
        "jacobian.math.algebra.trigonometric_rational.operations.cancel_common_factor",
        _forbidden,
    )
    scaled = _scaled_sine(10**3000)
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                "denominator": {
                    "kind": "DIVIDE",
                    "numerator": scaled,
                    "denominator": scaled,
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.numerator.terms[0].exponents == (0,)
    assert result.denominator.terms[0].exponents == (0,)
    assert {term.exponents for term in result.denominator_nonzero.terms} == {
        (0,),
        (2,),
    }
    assert all(
        abs(term.coefficient.real.num) < 10
        and abs(term.coefficient.imaginary.num) < 10
        for term in result.denominator_nonzero.terms
    )


def test_equivalent_locus_factors_are_deduplicated_before_combining() -> None:
    sine = {"kind": "SINE", "angle": {"coefficients": [3000]}}
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                "denominator": {
                    "kind": "DIVIDE",
                    "numerator": sine,
                    "denominator": sine,
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert result.numerator.terms[0].exponents == (0,)
    assert result.denominator.terms[0].exponents == (0,)
    assert {term.exponents for term in result.denominator_nonzero.terms} == {
        (3000,),
        (-3000,),
    }


def test_reduced_denominator_support_is_bounded_before_gcd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("GCD worker started for an oversized reduced denominator")

    monkeypatch.setattr(
        "jacobian.math.algebra.trigonometric_rational.operations.cancel_common_factor",
        _forbidden,
    )
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x", "y"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {
                    "kind": "MULTIPLY",
                    "children": [
                        {"kind": "SINE", "angle": {"coefficients": [1, 0]}},
                        {"kind": "SINE", "angle": {"coefficients": [0, 1]}},
                    ],
                },
                "denominator": {
                    "kind": "MULTIPLY",
                    "children": [
                        {"kind": "SINE", "angle": {"coefficients": [65, 0]}},
                        {"kind": "SINE", "angle": {"coefficients": [0, 65]}},
                    ],
                },
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError, match="support"):
        normalize_trigonometric_rational(request)
