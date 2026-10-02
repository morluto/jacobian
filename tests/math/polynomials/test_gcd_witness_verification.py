"""GCD verification checks the relation, not one extended-Euclidean witness."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._models import (
    PolynomialBezoutIdentity,
    PolynomialGcdResult,
)
from jacobian.math.polynomials.operations import verify_polynomial_gcd
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    coefficients: tuple[int | Fraction, ...], axis: str = "x"
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=(axis,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(value)),
                    exponents=(degree,),
                )
                for degree, value in reversed(tuple(enumerate(coefficients)))
                if value
            )
        ),
    )


def _claim(
    left: tuple[int | Fraction, ...],
    right: tuple[int | Fraction, ...],
    gcd: tuple[int | Fraction, ...],
    s: tuple[int | Fraction, ...],
    t: tuple[int | Fraction, ...],
) -> PolynomialGcdResult:
    return PolynomialGcdResult(
        left=_poly(left),
        right=_poly(right),
        gcd=_poly(gcd),
        bezout=PolynomialBezoutIdentity(
            left_multiplier=_poly(s), right_multiplier=_poly(t)
        ),
    )


@pytest.mark.parametrize(
    "claim",
    [
        _claim((-1, 0, 1), (-1, 1), (-1, 1), (1,), (0, -1)),
        _claim((-1, 0, 1), (-1, 1), (-1, 1), (2, 1), (-1, -3, -1)),
        _claim(
            (1, 0, 1),
            (1, 1),
            (1,),
            (Fraction(3, 2), 1),
            (Fraction(-1, 2), Fraction(-1, 2), -1),
        ),
        _claim((2, 2), (), (1, 1), (Fraction(1, 2),), (0, 1)),
        _claim((), (2, 2), (1, 1), (0, 1), (Fraction(1, 2),)),
    ],
)
def test_accepts_alternative_exact_bezout_witnesses(claim: PolynomialGcdResult) -> None:
    decoded = PolynomialGcdResult.model_validate_json(claim.model_dump_json())
    assert verify_polynomial_gcd(decoded)


@pytest.mark.parametrize(
    "claim",
    [
        _claim((-1, 0, 1), (-1, 1), (-1, 1), (), ()),
        _claim((-1, 0, 1), (-1, 1), (-1, 0, 1), (1,), ()),
        _claim((-1, 0, 1), (-1, 1), (-2, 2), (2,), (0, -2)),
        _claim((0, 0, 1), (-1, 1), (-1, 1), (1,), (0, -1)),
        _claim((), (), (), (), ()),
    ],
)
def test_rejects_false_source_bound_gcd_relations(claim: PolynomialGcdResult) -> None:
    assert not verify_polynomial_gcd(claim)


def test_rejects_mismatched_witness_ring() -> None:
    claim = _claim((-1, 0, 1), (-1, 1), (-1, 1), (1,), (0, -1))
    wrong_ring = claim.model_copy(
        update={
            "bezout": claim.bezout.model_copy(
                update={"left_multiplier": _poly((1,), "y")}
            )
        }
    )
    assert not verify_polynomial_gcd(wrong_ring)


def test_accepts_syzygy_shift_beyond_producer_degree_limit() -> None:
    # F=(x+1)G, S=x^600, T=1-(x+1)x^600.
    s = (0,) * 600 + (1,)
    t = (1,) + (0,) * 599 + (-1, -1)
    assert verify_polynomial_gcd(_claim((-1, 0, 1), (-1, 1), (-1, 1), s, t))


@pytest.mark.parametrize("linear_divisor", (False, True))
def test_accepts_maximum_width_scalar_content(linear_divisor: bool) -> None:
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

    content = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    source = (content, content) if linear_divisor else (0, content)
    divisor = (1, 1) if linear_divisor else (0, 1)
    claim = _claim(source, (), divisor, (Fraction(1, content),), ())
    assert verify_polynomial_gcd(claim)


def test_accepts_maximum_width_canceling_alternative_witness() -> None:
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

    content = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    assert verify_polynomial_gcd(_claim((1,), (1,), (1,), (content,), (1 - content,)))


def test_rejects_forged_native_candidate_fields() -> None:
    claim = _claim((-1, 0, 1), (-1, 1), (-1, 1), (1,), (0, -1))
    assert not verify_polynomial_gcd(claim.model_copy(update={"normalization": "RAW"}))
    assert not verify_polynomial_gcd(claim.model_copy(update={"gcd": None}))
    term = claim.bezout.left_multiplier.polynomial.terms[0]
    forged = term.model_copy(
        update={"coefficient": CanonicalRational.model_construct(num=2, den=2)}
    )
    multiplier = claim.bezout.left_multiplier.model_copy(
        update={"polynomial": SparseRationalPolynomial.model_construct(terms=(forged,))}
    )
    assert not verify_polynomial_gcd(
        claim.model_copy(
            update={
                "bezout": claim.bezout.model_copy(
                    update={"left_multiplier": multiplier}
                )
            }
        )
    )


def test_candidate_growth_refusal_is_not_a_false_verdict() -> None:
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.catalog.models import OperationResourceAdmissionError

    content = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    candidate = _claim((content,) * 4, (1,), (1,), (content,) * 4, (1,))
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_polynomial_gcd(candidate)
    assert error.value.errors()[0]["type"] == "polynomial.gcd_verification_budget"


def test_accepts_sparse_maximum_degree_divisibility() -> None:
    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    source = (0,) * (MAX_POLYNOMIAL_EXPONENT - 1) + (1, 1)
    assert verify_polynomial_gcd(_claim(source, (1, 1), (1, 1), (), (1,)))


def test_accepts_canceling_products_beyond_carrier_exponent() -> None:
    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    s = (1,) + (0,) * (MAX_POLYNOMIAL_EXPONENT - 1) + (1,)
    t = (0,) * MAX_POLYNOMIAL_EXPONENT + (-1,)
    assert verify_polynomial_gcd(_claim((0, 1), (0, 1), (0, 1), s, t))


def test_zero_multiplier_keeps_distinct_denominators_sparse() -> None:
    from sympy import primerange

    coefficients = tuple(Fraction(1, int(prime) ** 100) for prime in primerange(2, 140))
    assert verify_polynomial_gcd(_claim(coefficients, (1,), (1,), (), (1,)))


def test_divisibility_growth_refusal_is_not_a_false_verdict() -> None:
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.catalog.models import OperationResourceAdmissionError

    content = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    claim = _claim((content,) + (0,) * 9 + (1,), (content, 1), (content, 1), (), (1,))
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_polynomial_gcd(claim)
    assert error.value.errors()[0]["type"] == "polynomial.gcd_verification_budget"


def test_accepts_monic_polynomial_subtype_at_native_boundary() -> None:
    from jacobian.math.polynomials.values import MonicPolynomial

    claim = _claim((-1, 1), (-1, 1), (-1, 1), (), (1,))
    monic = MonicPolynomial.model_validate(claim.gcd.model_dump())
    specialized = PolynomialGcdResult(
        left=monic, right=monic, gcd=monic, bezout=claim.bezout
    )
    assert verify_polynomial_gcd(specialized)
