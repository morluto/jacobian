import time

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
        abs(term.coefficient.real.num) < 10 and abs(term.coefficient.imaginary.num) < 10
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
        abs(term.coefficient.real.num) < 10 and abs(term.coefficient.imaginary.num) < 10
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


def test_coupled_denominator_reduction_reports_typed_resource_admission() -> None:
    """A reduced quotient that exceeds the envelope must fail with a typed error.

    ``sin(65x)sin(65y)/(sin(x)sin(y)cos(x+y))`` cancels to a numerator with
    4225 terms, which is inside the preflight estimate but beyond the admitted
    output support. The reduction must surface the typed resource-admission
    error rather than an untyped model-validation failure.
    """

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
                        {
                            "kind": "COSINE",
                            "angle": {"coefficients": [1, 1]},
                        },
                    ],
                },
            },
        }
    )
    with pytest.raises(OperationResourceAdmissionError, match="expansion"):
        normalize_trigonometric_rational(request)


def test_high_degree_sparse_denominator_reduction_stays_within_deadline() -> None:
    """A sparse high-degree quotient must reduce through the fast backend.

    ``sin(4096x)/(cos(x)+2sin(x))`` shares the shifted support of
    ``sin(4096x)/sin(x)`` but does not cancel; its reduced numerator exceeds the
    output envelope. The reduction must return the typed admission error well
    inside the worker lease instead of exhausting the deadline.
    """

    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [4096]}},
                "denominator": {
                    "kind": "ADD",
                    "children": [
                        {"kind": "COSINE", "angle": {"coefficients": [1]}},
                        {
                            "kind": "MULTIPLY",
                            "children": [
                                {"kind": "LITERAL", "value": {"num": 2, "den": 1}},
                                {
                                    "kind": "SINE",
                                    "angle": {"coefficients": [1]},
                                },
                            ],
                        },
                    ],
                },
            },
        }
    )
    started = time.monotonic()
    with pytest.raises(OperationResourceAdmissionError, match="expansion"):
        normalize_trigonometric_rational(request)
    assert time.monotonic() - started < 5.0


def test_univariate_lattice_stride_still_admits_after_fast_backend() -> None:
    """The fast reduction must not regress the accepted ``sin(4096x)/sin(x)``."""

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


def test_reduced_canonical_exponent_is_admitted_before_result_construction() -> None:
    """``sin(4096x)/sin(4095x)`` reduces past the output exponent envelope.

    The reduced numerator reaches exponent 8189, so the request must fail with
    the typed resource-admission error instead of constructing thousands of
    quotient terms and failing model validation afterwards.
    """

    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [4096]}},
                "denominator": {"kind": "SINE", "angle": {"coefficients": [4095]}},
            },
        }
    )
    started = time.monotonic()
    with pytest.raises(OperationResourceAdmissionError, match="expansion"):
        normalize_trigonometric_rational(request)
    assert time.monotonic() - started < 5.0


def test_gaussian_gcd_uses_the_field_relation() -> None:
    """cos(2x)/(sin(x)+cos(x)) shares z^2+1-type factors over QQ(i).

    The flint QQ[z,I] path cannot see a factor that needs ``I^2 = -1``, so the
    worker must verify the candidate over QQ(i) and fall back to the ring path.
    The reduced quotient is real; sample it at x = 1/3 against the direct
    trigonometric value.
    """
    import sympy

    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "COSINE", "angle": {"coefficients": [2]}},
                "denominator": {
                    "kind": "ADD",
                    "children": [
                        {"kind": "SINE", "angle": {"coefficients": [1]}},
                        {"kind": "COSINE", "angle": {"coefficients": [1]}},
                    ],
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    z = sympy.Symbol("z")
    num = sum(
        (
            sympy.Rational(term.coefficient.real.as_fraction())
            + sympy.I * sympy.Rational(term.coefficient.imaginary.as_fraction())
        )
        * z ** term.exponents[0]
        for term in result.numerator.terms
    )
    den = sum(
        (
            sympy.Rational(term.coefficient.real.as_fraction())
            + sympy.I * sympy.Rational(term.coefficient.imaginary.as_fraction())
        )
        * z ** term.exponents[0]
        for term in result.denominator.terms
    )
    value = complex(
        sympy.N((num / den).subs(z, sympy.exp(sympy.I * sympy.Rational(1, 3))), 30)
    )
    direct = complex(
        sympy.N(
            sympy.cos(sympy.Rational(2, 3))
            / (sympy.sin(sympy.Rational(1, 3)) + sympy.cos(sympy.Rational(1, 3))),
            30,
        )
    )
    assert abs(value - direct) < 1e-9
    assert abs(value.imag) < 1e-9


def test_imaginary_flint_terms_keep_their_real_component() -> None:
    """A quotient term ``I*z^e`` must not be folded into the real component."""
    from flint import fmpq_mpoly_ctx

    from jacobian.math.algebra.trigonometric_rational._laurent_gcd_worker import (
        _flint_to_payload,
    )

    ctx = fmpq_mpoly_ctx.get(["z", "I"])
    z, imaginary = ctx.gens()
    payload = _flint_to_payload(z**2 + imaginary * z, 0)
    pairs = {
        support[0]: (real, imag)
        for support, real, imag in zip(
            payload["supports"],
            payload["real_numerators"],
            payload["imag_numerators"],
            strict=True,
        )
    }
    assert pairs[2] == ("1", "0")
    assert pairs[1] == ("0", "1")


def test_locus_union_retains_the_larger_zero_set() -> None:
    """A locus factor dividing another must not replace it in the union.

    The union of zero sets is represented by the factor with the larger zero
    set: ``(1-z^2)`` has zeros at z = +-1, so ``(1-z^2)`` must be retained over
    the smaller ``(1-z)``.
    """
    from fractions import Fraction

    from jacobian.math.algebra.trigonometric_rational.operations import _combine_loci

    gaussian = (Fraction(1), Fraction())
    larger = {
        (2,): (-gaussian[0], gaussian[1]),
        (0,): gaussian,
    }
    smaller = {
        (1,): (-gaussian[0], gaussian[1]),
        (0,): gaussian,
    }
    combined = _combine_loci((larger, smaller), 1)
    assert set(combined) == set(larger)


def test_locus_divisibility_uses_the_laurent_ring() -> None:
    """sin(3000x) divides sin(3000x)*sin(x) in the Laurent ring.

    The reduced zero-set cover keeps the larger factor, so 1/(P/(P*sin(x)))
    with P = sin(3000x) is admitted instead of rejecting the product of both
    recorded factors at exponent 6001.
    """
    request = TrigonometricRationalSource.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                "denominator": {
                    "kind": "DIVIDE",
                    "numerator": {"kind": "SINE", "angle": {"coefficients": [3000]}},
                    "denominator": {
                        "kind": "MULTIPLY",
                        "children": [
                            {"kind": "SINE", "angle": {"coefficients": [3000]}},
                            {"kind": "SINE", "angle": {"coefficients": [1]}},
                        ],
                    },
                },
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.denominator_nonzero.terms) == 4
