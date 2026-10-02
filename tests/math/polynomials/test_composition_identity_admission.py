"""Identity substitution preserves admitted canonical polynomial outputs."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.polynomials.maps._models import CompositionResult
from jacobian.math.polynomials.maps.operations import compose_polynomials
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variable: str, terms: Mapping[int, int | Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=(variable,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=(degree,),
                )
                for degree, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _compose(outer: RationalPolynomial, inner: RationalPolynomial) -> CompositionResult:
    return compose_polynomials(
        outer,
        inner,
        outer_variable=outer.variables[0],
        inner_variable=inner.variables[0],
    )


@pytest.mark.parametrize("coefficient_growth", (False, True))
def test_composition_output_survives_identity_readmission(
    coefficient_growth: bool,
) -> None:
    outer = _polynomial("u", {64: 1})
    inner = (
        _polynomial("x", {1: 10**127})
        if coefficient_growth
        else _polynomial("x", {2: 1})
    )
    produced = _compose(outer, inner)
    expected = _polynomial("x", {64: 10**8128} if coefficient_growth else {128: 1})
    assert produced.polynomial == expected

    decoded = CompositionResult.model_validate_json(produced.model_dump_json())
    retained = _compose(decoded.polynomial, _polynomial("x", {1: 1}))
    assert retained.polynomial == expected
    assert CompositionResult.model_validate_json(retained.model_dump_json()) == retained


@pytest.mark.parametrize("side", ("outer", "inner"))
@pytest.mark.parametrize(
    "terms", ({}, {0: Fraction(-7, 13)}, {128: 3, 2: Fraction(-2, 7), 0: 9})
)
def test_identity_preserves_zero_constants_signs_and_explicit_axis(
    side: str, terms: dict[int, int | Fraction]
) -> None:
    outer = _polynomial("u", {1: 1} if side == "outer" else terms)
    inner = _polynomial("x", terms if side == "outer" else {1: 1})
    result = _compose(outer, inner)
    assert result.polynomial == _polynomial("x", terms)
    assert result.polynomial.variables == ("x",)
    assert CompositionResult.model_validate(result.model_dump()) == result
    assert CompositionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("side", ("outer", "inner"))
def test_identity_retains_full_canonical_exponent_and_scalar_boundary(
    side: str,
) -> None:
    magnitude = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    coefficient = Fraction(magnitude + 1, magnitude - 1)
    source = _polynomial("x", {MAX_POLYNOMIAL_EXPONENT: coefficient})
    identity = _polynomial("x", {1: 1})
    result = (
        _compose(identity, source) if side == "outer" else _compose(source, identity)
    )
    assert result.polynomial == source
    assert CompositionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("side", ("outer", "inner"))
@pytest.mark.parametrize("count", (256, 257, MAX_POLYNOMIAL_TERMS))
def test_identity_uses_retained_support_budget(side: str, count: int) -> None:
    source = _polynomial("x", dict.fromkeys(range(count), 1))
    identity = _polynomial("x", {1: 1})
    outer, inner = (identity, source) if side == "outer" else (source, identity)
    result = _compose(outer, inner)
    assert result.polynomial == source
    assert CompositionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("side", ("outer", "inner"))
def test_identity_still_checks_declared_variable(side: str) -> None:
    source = _polynomial("x", {128: 1})
    identity = _polynomial("x", {1: 1})
    outer, inner = (identity, source) if side == "outer" else (source, identity)
    with pytest.raises(OperationDomainValidationError):
        compose_polynomials(outer, inner, outer_variable="u", inner_variable="x")
    with pytest.raises(OperationDomainValidationError):
        compose_polynomials(outer, inner, outer_variable="x", inner_variable="u")


@pytest.mark.parametrize(
    ("outer_terms", "inner_terms"),
    (({64: 1}, {3: 1}), ({128: 1}, {1: 2}), ({2: 10**128}, {1: 1, 0: 1})),
)
def test_nonidentity_composition_keeps_expansion_envelope(
    outer_terms: dict[int, int], inner_terms: dict[int, int]
) -> None:
    with pytest.raises(OperationDomainValidationError):
        _compose(_polynomial("u", outer_terms), _polynomial("x", inner_terms))


@pytest.mark.parametrize("side", ("outer", "inner"))
@pytest.mark.parametrize(
    "mutation", ("coefficient", "exponent", "order", "axis", "domain")
)
def test_identity_cannot_return_a_forged_native_carrier(
    side: str, mutation: str
) -> None:
    source = _polynomial("x", {2: 1, 0: 1})
    first, second = source.polynomial.terms
    if mutation == "coefficient":
        first = first.model_copy(
            update={"coefficient": CanonicalRational.model_construct(num=2, den=2)}
        )
        source = source.model_copy(
            update={
                "polynomial": SparseRationalPolynomial.model_construct(
                    terms=(first, second)
                )
            }
        )
    elif mutation == "exponent":
        first = first.model_copy(update={"exponents": (-1,)})
        source = source.model_copy(
            update={
                "polynomial": SparseRationalPolynomial.model_construct(
                    terms=(first, second)
                )
            }
        )
    elif mutation == "order":
        source = source.model_copy(
            update={
                "polynomial": SparseRationalPolynomial.model_construct(
                    terms=(second, first)
                )
            }
        )
    elif mutation == "axis":
        source = source.model_copy(update={"variables": ("x", "y")})
    else:
        source = source.model_copy(update={"domain": "ZZ"})
    identity = _polynomial("x", {1: 1})
    outer, inner = (identity, source) if side == "outer" else (source, identity)
    with pytest.raises(OperationDomainValidationError) as error:
        _compose(outer, inner)
    assert error.value.errors()[0]["type"] == "polynomial.composition_operand"


def test_identity_does_not_allocate_a_symbolic_backend_polynomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials.maps import operations

    conversions = 0
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    original = rational_polynomial_to_sympy

    def count_conversion(value: RationalPolynomial) -> Any:
        nonlocal conversions
        conversions += 1
        return original(value)

    monkeypatch.setattr(operations, "rational_polynomial_to_sympy", count_conversion)
    source = _polynomial("u", {MAX_POLYNOMIAL_EXPONENT: 10**8128})
    result = _compose(source, _polynomial("x", {1: 1}))
    assert result.polynomial == _polynomial("x", {MAX_POLYNOMIAL_EXPONENT: 10**8128})
    assert conversions == 0


@pytest.mark.parametrize(
    "mutation", ("terms", "variables", "exponents", "coefficient", "scalar", "domain")
)
def test_forged_native_containers_are_bounded_before_materialization(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source = _polynomial("x", {2: 1})
    term = source.polynomial.terms[0]
    if mutation == "domain":
        source = source.model_copy(update={"domain": {"unexpected": [1] * 5000}})
    elif mutation == "variables":
        source = source.model_copy(update={"variables": ("x",) * 9})
    else:
        if mutation == "terms":
            terms = (term,) * (MAX_POLYNOMIAL_TERMS + 1)
        else:
            if mutation == "exponents":
                term = term.model_copy(update={"exponents": (1,) * 9})
            elif mutation == "coefficient":
                term = term.model_copy(
                    update={"coefficient": {"num": [1] * 5000, "den": 1}}
                )
            else:
                coefficient = CanonicalRational.model_construct(
                    num=1 << (4 * MAX_CANONICAL_RATIONAL_DIGITS), den=1
                )
                term = term.model_copy(update={"coefficient": coefficient})
            terms = (term,)
        source = source.model_copy(
            update={"polynomial": SparseRationalPolynomial.model_construct(terms=terms)}
        )
    identity = _polynomial("x", {1: 1})
    dumps = 0
    original = RationalPolynomial.model_dump

    def count_dump(
        self: RationalPolynomial, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal dumps
        dumps += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RationalPolynomial, "model_dump", count_dump)
    with pytest.raises(OperationDomainValidationError) as error:
        compose_polynomials(source, identity, outer_variable="x", inner_variable="x")
    assert error.value.errors()[0]["type"] == "polynomial.composition_operand"
    assert dumps == 0
