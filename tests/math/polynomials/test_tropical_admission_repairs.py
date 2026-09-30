"""Regressions for the tropical native admission and growth envelopes.

Each case pairs a defect fixed in this batch with a negative control showing
the enclosing bound still rejects the genuinely over-limit request.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import VectorProjectivizeRequest
from jacobian.math.polynomials.tropical._tools import compute_vector_projectivize
from jacobian.math.polynomials.tropical.operations import (
    tropical_polynomial_power,
    tropical_polynomial_univariate_roots,
    tropical_polynomial_univariate_split_form,
    tropical_vector_projectivize,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_POLYNOMIAL_TERMS,
    MAX_TROPICAL_ROOT_CROSSOVER_PAIRS,
    MAX_TROPICAL_SCALAR_DIGITS,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
)

MIN_PLUS_QQ = TropicalSemiring(convention="MIN_PLUS", base="QQ")


def _code(error: OperationDomainValidationError) -> str:
    """Return the owner diagnostic code carried by a domain rejection."""

    return str(error.errors()[0]["type"])


def _scalar(numerator: int, denominator: int = 1) -> TropicalScalar:
    return TropicalScalar(
        semiring=MIN_PLUS_QQ,
        kind="FINITE",
        value=CanonicalRational.from_fraction(Fraction(numerator, denominator)),
    )


def _vector(
    axis: tuple[str, ...], entries: tuple[TropicalScalar, ...]
) -> TropicalVector:
    return TropicalVector(semiring=MIN_PLUS_QQ, axis=axis, entries=entries)


def _monomial(exponent: int, coefficient_digits: int) -> TropicalPolynomial:
    return TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=(
            TropicalPolynomialTerm(
                exponents=(exponent,), coefficient=_scalar(10**coefficient_digits)
            ),
        ),
    )


# --- projectivize admits the whole vector contract -------------------------


def test_projectivize_rejects_forged_semiring_convention() -> None:
    forged = TropicalVector.model_construct(
        semiring=TropicalSemiring.model_construct(
            convention="SIDEWAYS", base="NONSENSE"
        ),
        axis=(),
        entries=(),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.semiring_invalid"


def test_projectivize_rejects_forged_axis_label() -> None:
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ, axis=("",), entries=(_scalar(3),)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.vector_axis_label"


def test_projectivize_rejects_non_tuple_container() -> None:
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ, axis=["a"], entries=[_scalar(3)]
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.vector_shape"


def test_projectivize_normal_vector_still_projects() -> None:
    kind, representative, _ = tropical_vector_projectivize(
        _vector(("a", "b"), (_scalar(2), _scalar(5)))
    )
    assert kind == "PROJECTIVIZED"
    assert representative is not None
    assert [entry.value.as_fraction() for entry in representative.entries] == [  # type: ignore[union-attr]
        Fraction(0),
        Fraction(3),
    ]


def test_projectivize_empty_vector_has_no_projective_class() -> None:
    kind, representative, translation = tropical_vector_projectivize(_vector((), ()))
    assert kind == "NO_PROJECTIVE_CLASS"
    assert representative is None
    assert translation is None


# Negative controls for the projectivize admission.


def test_projectivize_still_enforces_vector_dimension() -> None:
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ,
        axis=tuple(f"x{index}" for index in range(129)),
        entries=tuple(_scalar(1) for _ in range(129)),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.vector_dimension"


def test_projectivize_still_rejects_duplicate_axis() -> None:
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ, axis=("a", "a"), entries=(_scalar(1), _scalar(2))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.vector_shape"


def test_projectivize_still_rejects_entry_semiring_mismatch() -> None:
    entry = TropicalScalar.model_construct(
        semiring=TropicalSemiring(convention="MAX_PLUS", base="ZZ"),
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(1, 1),
    )
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ, axis=("a",), entries=(entry,)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tropical_vector_projectivize(forged)
    assert _code(error.value) == "tropical.semiring_mismatch"


def test_projectivize_forged_label_reports_domain_error_through_the_tool() -> None:
    """The catalog boundary must surface a domain error, not a carrier leak."""
    forged = TropicalVector.model_construct(
        semiring=MIN_PLUS_QQ, axis=("",), entries=(_scalar(3),)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_vector_projectivize(VectorProjectivizeRequest(vector=forged))
    assert _code(error.value) == "tropical.vector_axis_label"


# --- projectivize normalises exact cancellation ----------------------------


def test_projectivize_admits_exact_cancellation_near_the_envelope() -> None:
    """A pivot whose translation is exactly zero must not be rejected."""
    n = 10**5000
    kind, representative, _ = tropical_vector_projectivize(
        _vector(("a",), (_scalar(n, n - 1),))
    )
    assert kind == "PROJECTIVIZED"
    assert representative is not None
    assert representative.entries[0].value.as_fraction() == 0  # type: ignore[union-attr]


def test_projectivize_still_rejects_genuine_growth() -> None:
    """A reduced result beyond the envelope must still be refused."""
    n = 10**8200
    with pytest.raises(OperationResourceAdmissionError):
        tropical_vector_projectivize(
            _vector(("a", "b"), (_scalar(1, 3), _scalar(n, n - 1)))
        )


def test_scalar_add_still_rejects_genuine_growth() -> None:
    from jacobian.math.polynomials.tropical.operations import tropical_scalar_add

    large = 10**8200
    with pytest.raises(OperationResourceAdmissionError):
        tropical_scalar_add(
            MIN_PLUS_QQ,
            TropicalScalar(
                semiring=MIN_PLUS_QQ,
                kind="FINITE",
                value=CanonicalRational.from_fraction(Fraction(1, 3)),
            ),
            _scalar(large, large - 1),
        )


# --- split form does not charge a quadratic crossover envelope -------------


def test_split_form_accepts_support_within_the_term_envelope() -> None:
    count = 363
    polynomial = TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(exponents=(index,), coefficient=_scalar(index + 1))
            for index in range(count)
        ),
    )
    assert count * (count - 1) // 2 > MAX_TROPICAL_ROOT_CROSSOVER_PAIRS
    assert count <= MAX_TROPICAL_POLYNOMIAL_TERMS
    result = tropical_polynomial_univariate_split_form(polynomial)
    assert len(result.terms) == count


def test_split_form_still_enforces_the_term_envelope() -> None:
    span = MAX_TROPICAL_POLYNOMIAL_TERMS + 5
    polynomial = TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=(
            TropicalPolynomialTerm(exponents=(0,), coefficient=_scalar(1)),
            TropicalPolynomialTerm(exponents=(span,), coefficient=_scalar(2)),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_univariate_split_form(polynomial)
    assert _code(error.value) == "tropical.split_form_terms"


# --- exponent one performs no coefficient arithmetic -----------------------


def test_power_one_reuses_admitted_coefficients() -> None:
    polynomial = _monomial(1, MAX_TROPICAL_SCALAR_DIGITS - 1)
    result = tropical_polynomial_power(polynomial, 1)
    assert result is polynomial


def test_power_two_still_rejects_genuine_growth() -> None:
    big = 10**8000
    polynomial = TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=(
            TropicalPolynomialTerm(exponents=(0,), coefficient=_scalar(big)),
            TropicalPolynomialTerm(exponents=(1,), coefficient=_scalar(big)),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_power(polynomial, 2)
    assert _code(error.value) == "tropical.polynomial_power_scalar_bound"


# --- monomials have no line intersections ----------------------------------


def test_monomial_roots_skip_the_crossover_digit_estimate() -> None:
    polynomial = _monomial(0, MAX_TROPICAL_SCALAR_DIGITS - 2)
    profile = tropical_polynomial_univariate_roots(polynomial)
    assert profile.kind == "FINITE_PROFILE"
    assert profile.roots == ()
    assert profile.source is polynomial


def test_two_term_roots_still_enforce_the_crossover_digit_bound() -> None:
    big = 10**8190
    polynomial = TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=(
            TropicalPolynomialTerm(exponents=(0,), coefficient=_scalar(big)),
            TropicalPolynomialTerm(exponents=(1,), coefficient=_scalar(big + 1)),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_univariate_roots(polynomial)
    assert _code(error.value) == "tropical.root_digit_bound"


def test_root_profile_still_enforces_the_crossover_pair_bound() -> None:
    count = 400
    polynomial = TropicalPolynomial(
        semiring=MIN_PLUS_QQ,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(exponents=(index,), coefficient=_scalar(index + 1))
            for index in range(count)
        ),
    )
    assert count * (count - 1) // 2 > MAX_TROPICAL_ROOT_CROSSOVER_PAIRS
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_univariate_roots(polynomial)
    assert _code(error.value) == "tropical.root_crossover_work"
