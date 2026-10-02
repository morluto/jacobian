"""Compact square-free sources use a degree-one maintained backend kernel."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import _sympy
from jacobian.math.polynomials._models import PolynomialSquareFreeDecompositionResult
from jacobian.math.polynomials.operations import (
    polynomial_square_free_decomposition,
    verify_polynomial_square_free_decomposition,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    terms: dict[tuple[int, ...], Fraction | int], variables: tuple[str, ...] = ("x",)
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(
                        Fraction(value).numerator, Fraction(value).denominator
                    ),
                    exponents=exponents,
                )
                for exponents, value in sorted(terms.items(), reverse=True)
            )
        ),
    )


@pytest.mark.parametrize("degree", (1, 64, 65, 256, 500, MAX_POLYNOMIAL_EXPONENT))
@pytest.mark.parametrize(
    "leading,constant", ((1, 1), (Fraction(-2, 3), Fraction(5, 7)))
)
def test_sparse_binomial_has_the_independently_known_decomposition(
    degree: int, leading: Fraction | int, constant: Fraction | int
) -> None:
    source = _polynomial({(degree,): leading, (0,): constant}, ("theta",))
    result = polynomial_square_free_decomposition(source)
    assert result.polynomial == result.reconstructed == source
    assert result.coefficient.as_fraction() == leading
    assert len(result.factors) == 1
    record = result.factors[0]
    assert record.multiplicity == 1
    assert record.factor == _polynomial(
        {(degree,): 1, (0,): Fraction(constant) / Fraction(leading)}, ("theta",)
    )
    # Independent sparse product identity: scale each coefficient. Together
    # with b != 0, gcd(a*x^n+b, a*n*x^(n-1))=1 proves square-freeness over QQ.
    assert (
        _polynomial(
            {
                term.exponents: result.coefficient.as_fraction()
                * term.coefficient.as_fraction()
                for term in record.factor.polynomial.terms
            },
            ("theta",),
        )
        == source
    )
    decoded = PolynomialSquareFreeDecompositionResult.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert decoded == result
    assert verify_polynomial_square_free_decomposition(decoded)


def test_compact_boundary_really_runs_only_the_degree_one_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    backend = _sympy.polynomial_square_free_decomposition

    def record_backend(source: Any) -> Any:
        calls.append(source.degree())
        return backend(source)

    monkeypatch.setattr(_sympy, "polynomial_square_free_decomposition", record_backend)
    source = _polynomial({(MAX_POLYNOMIAL_EXPONENT,): 1, (0,): -2})
    result = polynomial_square_free_decomposition(source)
    assert calls == [1]
    assert result.reconstructed == source
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    assert calls == [1]  # Serialization and strict decoding are structural only.


def test_binomial_coefficient_normalization_has_bounded_growth() -> None:
    large = 10**256 - 1
    source = _polynomial({(MAX_POLYNOMIAL_EXPONENT,): Fraction(1, large), (0,): large})
    result = polynomial_square_free_decomposition(source)
    constant = result.factors[0].factor.polynomial.terms[1].coefficient
    assert constant.num == large**2 and constant.den == 1
    assert len(str(constant.num)) == 512
    assert result.reconstructed == source
    assert verify_polynomial_square_free_decomposition(result)


@pytest.mark.parametrize(
    "source",
    (
        _polynomial({(65,): 1, (1,): 1}),
        _polynomial({(65,): 1, (1,): 1, (0,): 1}),
        _polynomial({(65, 0): 1, (0, 0): 1}, ("y", "x")),
        _polynomial({(0, 65): 1, (0, 0): 1}, ("y", "x")),
        _polynomial({(65, 1): 1, (0, 0): 1}, ("y", "x")),
    ),
)
def test_non_binomial_general_degree_keeps_conservative_admission(
    source: RationalPolynomial,
) -> None:
    with pytest.raises(OperationDomainValidationError):
        polynomial_square_free_decomposition(source)


@pytest.mark.parametrize(
    "source",
    (
        _polynomial({(64,): 1, (1,): 1}),
        _polynomial({(64,): 1, (1,): 1, (0,): 1}),
        _polynomial({(64, 0): 1, (0, 0): 1}, ("y", "x")),
        _polynomial({(0, 64): 1, (0, 0): 1}, ("y", "x")),
    ),
)
def test_existing_general_boundary_preserves_the_declared_axes(
    source: RationalPolynomial,
) -> None:
    result = polynomial_square_free_decomposition(source)
    assert result.polynomial == result.reconstructed == source
    assert all(record.factor.variables == source.variables for record in result.factors)
    assert verify_polynomial_square_free_decomposition(result)


def test_multiplicity_limit_is_separate_from_source_degree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = polynomial_square_free_decomposition(_polynomial({(64,): 1}))
    assert result.factors[0].multiplicity == 64
    assert result.factors[0].factor == _polynomial({(1,): 1})
    calls: list[int] = []
    backend = _sympy.polynomial_square_free_decomposition

    def record_backend(source: Any) -> Any:
        calls.append(source.degree())
        return backend(source)

    monkeypatch.setattr(_sympy, "polynomial_square_free_decomposition", record_backend)
    for source in (_polynomial({(65,): 1}), _polynomial({(1, 65): 1}, ("y", "x"))):
        with pytest.raises(OperationResourceAdmissionError, match="multiplicity"):
            polynomial_square_free_decomposition(source)
    assert not calls  # Reject the unrepresentable multiplicity before the kernel.


@pytest.mark.parametrize("terms", ({}, {(0,): Fraction(-2, 3)}))
def test_degenerate_sources_remain_canonical(terms: dict[tuple[int, ...], Any]) -> None:
    source = _polynomial(terms)
    result = polynomial_square_free_decomposition(source)
    assert not result.factors
    assert result.polynomial == result.reconstructed == source
    assert verify_polynomial_square_free_decomposition(result)


@pytest.mark.parametrize(
    "field", ("polynomial", "factor", "reconstructed", "multiplicity")
)
def test_structural_decoding_does_not_accept_forged_mathematics(field: str) -> None:
    result = polynomial_square_free_decomposition(_polynomial({(256,): 1, (0,): 1}))
    payload = result.model_dump()
    changed = _polynomial({(256,): 1, (0,): 2}).model_dump()
    if field == "factor":
        payload["factors"][0]["factor"] = changed
    elif field == "multiplicity":
        payload["factors"][0]["multiplicity"] = 2
    else:
        payload[field] = changed
    forged = PolynomialSquareFreeDecompositionResult.model_validate(
        payload, strict=True
    )
    assert not verify_polynomial_square_free_decomposition(forged)


def test_unrepresentable_multiplicity_stays_a_structural_decode_error() -> None:
    result = polynomial_square_free_decomposition(_polynomial({(65,): 1, (0,): 1}))
    payload = result.model_dump()
    payload["factors"][0]["multiplicity"] = 65
    with pytest.raises(ValidationError):
        PolynomialSquareFreeDecompositionResult.model_validate(payload, strict=True)


def test_verifier_preserves_multiplicity_resource_refusal() -> None:
    result = polynomial_square_free_decomposition(_polynomial({(65,): 1, (0,): 1}))
    payload = result.model_dump()
    payload["polynomial"] = _polynomial({(65,): 1}).model_dump()
    claim = PolynomialSquareFreeDecompositionResult.model_validate(payload, strict=True)
    with pytest.raises(OperationResourceAdmissionError, match="multiplicity"):
        verify_polynomial_square_free_decomposition(claim)


@pytest.mark.parametrize("source", (None, 1, {}, "x^65+1"))
def test_native_source_type_refusal_is_typed(source: Any) -> None:
    with pytest.raises(OperationDomainValidationError, match="RationalPolynomial"):
        polynomial_square_free_decomposition(source)
