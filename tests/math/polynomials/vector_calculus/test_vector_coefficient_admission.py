"""Exact coefficient work and output admission for vector derivatives."""

from fractions import Fraction

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.polynomials.vector_calculus import operations
from jacobian.math.polynomials.vector_calculus._models import ScalarResult, VectorResult


def _polynomial(terms: dict[tuple[int, ...], int | Fraction]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
            )
        ),
    )


def test_vector_consumers_accept_canonical_height_linear_and_cancellation() -> None:
    coefficient = 10**MAX_CANONICAL_RATIONAL_DIGITS - 1
    zero = _polynomial({})
    source = (_polynomial({(1, 0, 0): coefficient}), zero, zero)
    result = operations.divergence(source)
    assert result.result == _polynomial({(0, 0, 0): coefficient})
    assert ScalarResult.model_validate_json(result.model_dump_json()) == result
    assert operations.curl(source).components == (zero, zero, zero)

    # Opposite signs cancel even when an individual partial exceeds the
    # canonical output bound. A bound before signed collection is too coarse.
    source = (
        _polynomial({(1, 2, 0): coefficient}),
        _polynomial({(2, 1, 0): coefficient}),
        zero,
    )
    result_curl = operations.curl(source)
    assert result_curl.components == (zero, zero, zero)
    assert (
        VectorResult.model_validate_json(result_curl.model_dump_json()) == result_curl
    )


def test_vector_consumer_collects_large_distinct_denominators_exactly() -> None:
    denominator = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 2)
    source = (
        _polynomial({(1, 0, 0): Fraction(1, denominator)}),
        _polynomial({(0, 1, 0): Fraction(-1, 2 * denominator)}),
        _polynomial({}),
    )
    result = operations.divergence(source)
    assert result.result == _polynomial({(0, 0, 0): Fraction(1, 2 * denominator)})


@pytest.mark.parametrize("kind", ["product", "denominator", "work"])
@pytest.mark.parametrize("operation", ["divergence", "curl"])
def test_vector_resource_refusal_precedes_symbolic_expansion_and_survives_verifier(
    kind: str, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    zero = _polynomial({})
    if kind == "product":
        exponents = (2, 0, 0) if operation == "divergence" else (0, 2, 0)
        source = (
            _polynomial({exponents: 10**MAX_CANONICAL_RATIONAL_DIGITS - 1}),
            zero,
            zero,
        )
    elif kind == "denominator":
        denominator = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS // 2)
        if operation == "divergence":
            source = (
                _polynomial({(1, 0, 0): Fraction(1, denominator)}),
                _polynomial({(0, 1, 0): Fraction(1, denominator + 1)}),
                zero,
            )
        else:
            source = (
                zero,
                _polynomial({(0, 0, 1): Fraction(-1, denominator + 1)}),
                _polynomial({(0, 1, 0): Fraction(1, denominator)}),
            )
    else:
        source = (
            _polynomial({(index, 0, 0): 10**23000 for index in range(3)}),
            zero,
            zero,
        )
    expected_code = (
        "polynomial_vector_calc.coefficient_work_budget"
        if kind == "work"
        else "polynomial_vector_calc.derivative_coefficient_bound"
    )

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("resource refusal must precede symbolic conversion")

    monkeypatch.setattr(operations, "_expressions", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError) as error:
        getattr(operations, operation)(source)
    assert error.value.errors()[0]["type"] == expected_code
    claim = (
        ScalarResult(source_components=source, result=zero)
        if operation == "divergence"
        else VectorResult(source_components=source, components=(zero, zero, zero))
    )
    with pytest.raises(OperationResourceAdmissionError) as verifier_error:
        getattr(operations, "verify_" + operation)(claim)
    assert verifier_error.value.errors()[0]["type"] == expected_code


@pytest.mark.parametrize("operation", ["divergence", "curl"])
def test_expanded_vector_claims_still_bind_sources_axes_and_result(
    operation: str,
) -> None:
    zero = _polynomial({})
    source = (_polynomial({(2, 1, 0): 10**200}), zero, zero)
    claim = getattr(operations, operation)(source)
    verify = getattr(operations, "verify_" + operation)
    decoded = type(claim).model_validate_json(claim.model_dump_json())
    assert verify(decoded)
    assert not verify(
        decoded.model_copy(update={"source_components": (zero, zero, zero)})
    )
    assert not verify(
        decoded.model_copy(update={"source_components": (zero, source[0], zero)})
    )
    renamed = source[0].model_copy(update={"variables": ("y", "x", "z")})
    assert not verify(
        decoded.model_copy(update={"source_components": (renamed, zero, zero)})
    )
    update = (
        {"result": zero}
        if isinstance(decoded, ScalarResult)
        else {"components": (zero, zero, zero)}
    )
    assert not verify(decoded.model_copy(update=update))


def test_scalar_source_coefficient_envelope_is_unchanged() -> None:
    source = _polynomial({(1, 0, 0): 10**128})
    for compute in (operations.gradient, operations.laplacian):
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute(source)
        assert exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.admission"
