"""Bézout witnesses own their output and complete bounded execution envelope."""

from fractions import Fraction
from random import Random
from typing import Any

import pytest
from sympy import Poly, Symbol

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import polynomial_gcd
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    coefficients: dict[int, Fraction | int], axis: str = "x"
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=(axis,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(c)),
                    exponents=(e,),
                )
                for e, c in sorted(coefficients.items(), reverse=True)
                if c
            )
        ),
    )


def _coefficients(value: RationalPolynomial) -> dict[int, Fraction]:
    return {t.exponents[0]: t.coefficient.as_fraction() for t in value.polynomial.terms}


def _assert_identity(result: Any) -> None:
    coefficients: dict[int, Fraction] = {}
    for multiplier, source in (
        (result.bezout.left_multiplier, result.left),
        (result.bezout.right_multiplier, result.right),
    ):
        for a, c in _coefficients(multiplier).items():
            for b, d in _coefficients(source).items():
                coefficients[a + b] = coefficients.get(a + b, Fraction()) + c * d
    assert {e: c for e, c in coefficients.items() if c} == _coefficients(result.gcd)
    assert result.gcd.polynomial.terms[0].coefficient.as_integer_ratio() == (1, 1)


@pytest.mark.parametrize("swap", (False, True))
@pytest.mark.parametrize("axis", ("x", "t"))
def test_large_representable_bezout_neighbor_is_exact_and_decodes(
    swap: bool, axis: str
) -> None:
    scalar = 10**255
    left, right = _poly({128: 1}, axis), _poly({1: -scalar, 0: 1}, axis)
    result = polynomial_gcd(right, left) if swap else polynomial_gcd(left, right)
    first, second = (
        (result.bezout.right_multiplier, result.bezout.left_multiplier)
        if swap
        else (result.bezout.left_multiplier, result.bezout.right_multiplier)
    )
    assert _coefficients(first) == {0: Fraction(scalar**128)}
    assert _coefficients(second) == {i: Fraction(scalar**i) for i in range(128)}
    _assert_identity(result)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_guaranteed_unrepresentable_witness_is_refused_before_conversion() -> None:
    scalar = 10**255
    assert scalar**129 >= 10**32768
    with pytest.raises(OperationResourceAdmissionError) as error:
        polynomial_gcd(_poly({129: 1}), _poly({1: -scalar, 0: 1}))
    assert error.value.errors()[0]["type"] == "polynomial.gcd.coefficient_height"


@pytest.mark.parametrize(
    "kind", ("self", "unit", "left_zero", "right_zero", "proportional", "linear")
)
def test_large_cheap_gcd_regimes_do_not_pay_for_a_dense_matrix(kind: str) -> None:
    source = _poly({500: 1, 1: 1, 0: 1})
    if kind == "self":
        left, right = source, source
    elif kind == "unit":
        left, right = source, _poly({0: Fraction(3, 7)})
    elif kind == "left_zero":
        left, right = _poly({}), source
    elif kind == "right_zero":
        left, right = source, _poly({})
    elif kind == "proportional":
        left, right = (
            _poly({500: Fraction(7, 11), 1: Fraction(7, 11), 0: Fraction(7, 11)}),
            _poly({500: Fraction(-21, 13), 1: Fraction(-21, 13), 0: Fraction(-21, 13)}),
        )
    else:
        left, right = _poly({500: 1}), _poly({1: 1, 0: -1})
    result = polynomial_gcd(left, right)
    _assert_identity(result)
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "kind", ("linear_remainder", "constant_remainder", "zero_remainder")
)
@pytest.mark.parametrize("swap", (False, True))
def test_support_proved_reduction_preserves_high_degree_accepted_cases(
    kind: str, swap: bool
) -> None:
    if kind == "linear_remainder":
        left, right = _poly({500: 1, 0: 1}), _poly({499: 1, 0: 2})
    elif kind == "constant_remainder":
        left, right = _poly({500: 10**255, 0: 1}), _poly({500: 10**255, 0: 2})
    else:
        left, right = _poly({500: 1}), _poly({400: 1})
    result = polynomial_gcd(right, left) if swap else polynomial_gcd(left, right)
    _assert_identity(result)
    assert _coefficients(result.gcd) == (
        {400: Fraction(1)} if kind == "zero_remainder" else {0: Fraction(1)}
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_generic_fraction_free_kernel_matches_independent_gcd_and_identity() -> None:
    random = Random(4368)
    x = Symbol("x")
    for _ in range(40):
        m, n = random.randrange(2, 8), random.randrange(2, 8)
        left = _poly(
            {
                i: Fraction(random.choice((-7, -2, 1, 3, 5)), random.randrange(1, 8))
                for i in range(m + 1)
            }
        )
        right = _poly(
            {
                i: Fraction(random.choice((-7, -2, 1, 3, 5)), random.randrange(1, 8))
                for i in range(n + 1)
            }
        )
        common = (
            Poly(x * x + 1, x, domain="QQ")
            if random.randrange(2)
            else Poly(1, x, domain="QQ")
        )
        left = rational_polynomial_from_sympy(
            rational_polynomial_to_sympy(left) * common, ("x",)
        )
        right = rational_polynomial_from_sympy(
            rational_polynomial_to_sympy(right) * common, ("x",)
        )
        expected = rational_polynomial_from_sympy(
            rational_polynomial_to_sympy(left).gcd(rational_polynomial_to_sympy(right)),
            ("x",),
        )
        result = polynomial_gcd(left, right)
        assert result.gcd == expected
        _assert_identity(result)
        assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "change",
    (
        {"domain": "ZZ"},
        {"variables": ["x"]},
        {"variables": ("bad name",)},
        {"polynomial": None},
    ),
)
def test_native_source_structure_is_not_laundered_by_a_shortcut(
    change: dict[str, object],
) -> None:
    source = _poly({5: 1, 0: 1}).model_copy(update=change)
    with pytest.raises(OperationDomainValidationError) as error:
        polynomial_gcd(source, source)
    assert error.value.errors()[0]["type"] == "polynomial.gcd.source"


def test_native_unreduced_scalar_remains_a_domain_error() -> None:
    source = _poly({5: 1})
    term = source.polynomial.terms[0].model_copy(
        update={"coefficient": CanonicalRational.model_construct(num=2, den=2)}
    )
    source = source.model_copy(
        update={"polynomial": source.polynomial.model_copy(update={"terms": (term,)})}
    )
    with pytest.raises(OperationDomainValidationError):
        polynomial_gcd(source, source)


@pytest.mark.parametrize(
    "limit,code",
    (
        ("MAX_BEZOUT_WORK", "work"),
        ("MAX_BEZOUT_COEFFICIENT_STORAGE_BITS", "storage"),
        ("MAX_BEZOUT_RESULT_DIGITS", "result_digits"),
    ),
)
def test_complete_private_budgets_remain_typed(
    monkeypatch: pytest.MonkeyPatch, limit: str, code: str
) -> None:
    from jacobian.math.polynomials import _bezout_kernel

    monkeypatch.setattr(_bezout_kernel, limit, 0)
    with pytest.raises(OperationResourceAdmissionError) as error:
        polynomial_gcd(_poly({2: 1, 0: 1}), _poly({0: 1}))
    assert error.value.errors()[0]["type"] == f"polynomial.gcd.{code}"


def test_scalar_bit_budget_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.polynomials import _bezout_kernel

    monkeypatch.setattr(_bezout_kernel, "MAX_BEZOUT_PRIVATE_BITS", 5)
    assert _bezout_kernel._Ledger().multiply(3, 4) == 12
    monkeypatch.setattr(_bezout_kernel, "MAX_BEZOUT_PRIVATE_BITS", 4)
    with pytest.raises(OperationResourceAdmissionError):
        _bezout_kernel._Ledger().multiply(3, 4)


def test_monic_producer_subtype_composes_without_ring_repair() -> None:
    from jacobian.math.polynomials.values import monic_polynomial_from_coefficients

    source = monic_polynomial_from_coefficients(
        (CanonicalRational(num=-1, den=1), CanonicalRational(num=1, den=1))
    )
    result = polynomial_gcd(source, source)
    assert result.left is source and result.right is source
    assert result.gcd.model_dump(mode="json") == source.model_dump(mode="json")
    _assert_identity(result)
    assert type(result).model_validate_json(result.model_dump_json()).model_dump(
        mode="json"
    ) == result.model_dump(mode="json")
