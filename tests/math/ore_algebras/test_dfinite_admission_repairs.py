"""Regressions for the D-finite series consumer boundaries.

Each repaired case is paired with a negative control showing the enclosing
bound or rejection is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

import jacobian.math.ore_algebras.operations as ore_operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    DFinitePowerSeries,
    DifferentialOreOperator,
    DifferentialOreTerm,
)
from jacobian.math.ore_algebras.operations import (
    differential_series_construct,
    differential_series_generate_prefix,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rf(terms: Sequence[tuple[Fraction | int, int]]) -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": {
                "terms": [
                    {
                        "coefficient": {
                            "num": Fraction(coefficient).numerator,
                            "den": Fraction(coefficient).denominator,
                        },
                        "exponents": [degree],
                    }
                    for coefficient, degree in terms
                ]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
        }
    )


def _operator(
    terms: Sequence[tuple[int, list[tuple[Fraction | int, int]]]],
) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "terms": [
                {"order": order, "coefficient": _rf(coefficients).model_dump()}
                for order, coefficients in sorted(terms, key=lambda item: item[0])
            ]
        }
    )


def _one() -> CanonicalRational:
    return CanonicalRational(num=1, den=1)


def _unit_term() -> RationalPolynomialTerm:
    return RationalPolynomialTerm.model_construct(exponents=(0,), coefficient=_one())


def _derivatives(*values: int) -> FiniteRationalSequence:
    return FiniteRationalSequence.model_validate(
        {"values": [{"num": value, "den": 1} for value in values]}
    )


def _prefix(series: DFinitePowerSeries, count: int) -> list[Fraction]:
    sequence = differential_series_generate_prefix(series, count)
    return [value.as_fraction() for value in sequence.values]


# --- a common x-adic factor does not make an equation singular --------------


def test_common_x_factor_is_cancelled_before_the_pivot_check() -> None:
    """x*D + x with f(0)=1 is (D+1)f=0, which is exp(-x)."""
    series = differential_series_construct(
        _operator([(0, [(1, 1)]), (1, [(1, 1)])]), _derivatives(1)
    )
    assert _prefix(series, 5) == [
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2),
        Fraction(-1, 6),
        Fraction(1, 24),
    ]


def test_a_higher_common_x_factor_is_also_cancelled() -> None:
    """x^3 * D^2 + x^3 = 0 reduces to D^2 + 1, so f'' = -1."""
    series = differential_series_construct(
        _operator([(0, [(1, 3)]), (2, [(1, 3)])]), _derivatives(1, 0)
    )
    # f(x) = 1 - x^2/2, so a_2 = -1/2 and the rest vanish.
    assert _prefix(series, 4) == [
        Fraction(1),
        Fraction(0),
        Fraction(-1, 2),
        Fraction(0),
    ]


# Negative control: an equation whose leading coefficient is singular at 0
# after removing the common factor has no Taylor series there.
def test_leading_coefficient_still_must_be_nonzero_at_the_center() -> None:
    forged = DFinitePowerSeries.model_construct(
        operator=_operator([(0, [(1, 0)]), (2, [(1, 1)])]),
        initial_derivatives=_derivatives(1, 0),
        center=0,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        differential_series_generate_prefix(forged, 3)
    assert (
        error.value.errors()[0]["type"]
        == "ore_algebra.dfinite_series_initial_value_problem"
    )


def test_singular_equation_is_refused_by_the_constructor() -> None:
    """x^65 * D + 1 is genuinely singular at 0 and must not be constructible."""
    with pytest.raises(OperationDomainValidationError):
        differential_series_construct(
            _operator([(0, [(1, 0)]), (1, [(1, 65)])]), _derivatives(1)
        )


# --- the consumer re-admits against the carrier's own envelope --------------


def test_wide_exponent_beyond_the_shift_budget_is_admitted() -> None:
    """D + x^70 is ordinary at 0 and within the representation envelope."""
    series = differential_series_construct(
        _operator([(0, [(1, 70)]), (1, [(1, 0)])]), _derivatives(1)
    )
    assert _prefix(series, 1) == [Fraction(1)]
    # f' = -x^70 f gives exp(-x^71/71), so only the constant survives.
    assert _prefix(series, 5) == [
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    ]


def test_wide_exponent_computes_the_expected_coefficient() -> None:
    """f' = -x f gives exp(-x^2/2), so the wide path is really evaluated."""
    series = differential_series_construct(
        _operator([(0, [(1, 1)]), (1, [(1, 0)])]), _derivatives(1)
    )
    assert _prefix(series, 5) == [
        Fraction(1),
        Fraction(0),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(1, 8),
    ]


# Negative control: the representation envelope itself is still enforced.
@pytest.mark.parametrize("limit_name", ["exponent", "digits"])
def test_forged_over_wide_coefficient_is_refused(limit_name: str) -> None:
    """A forged coefficient beyond the representation envelope never computes."""
    import jacobian.math.polynomials.values as values

    if limit_name == "exponent":
        payload: object = {
            "exponents": (values.MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT + 2,),
            "coefficient": _one(),
        }
    else:
        payload = {
            "exponents": (0,),
            "coefficient": CanonicalRational.model_construct(
                num=10 ** (values.MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS + 1),
                den=1,
            ),
        }
    wide = RationalFunction.model_construct(
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm.model_construct(**payload),)  # type: ignore[arg-type]
        ),
        denominator=SparseRationalPolynomial(terms=(_unit_term(),)),
    )
    forged = DFinitePowerSeries.model_construct(
        operator=DifferentialOreOperator.model_construct(
            variable="x",
            terms=(
                DifferentialOreTerm.model_construct(order=0, coefficient=wide),
                DifferentialOreTerm.model_construct(order=1, coefficient=_rf([(1, 0)])),
            ),
        ),
        initial_derivatives=_derivatives(1),
        center=0,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        differential_series_generate_prefix(forged, 1)
    # The carrier revalidation and the operation's own envelope check share the
    # same limits, so either may refuse first. Both refuse before any prefix.
    assert error.value.errors()[0]["type"] in {
        "ore_algebra.dfinite_prefix_request",
        "ore_algebra.differential_coefficient",
    }


def test_nonpolynomial_coefficients_are_still_refused() -> None:
    rational = RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [1]}]
            },
        }
    )
    forged = DFinitePowerSeries.model_construct(
        operator=DifferentialOreOperator.model_validate(
            {
                "terms": [
                    {"order": 0, "coefficient": _rf([(1, 0)]).model_dump()},
                    {"order": 1, "coefficient": rational.model_dump()},
                ]
            }
        ),
        initial_derivatives=_derivatives(1),
        center=0,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        differential_series_generate_prefix(forged, 3)
    assert (
        error.value.errors()[0]["type"]
        == "ore_algebra.dfinite_prefix_polynomial_coefficients"
    )


# --- the recurrence observes cancellation ----------------------------------


def test_recurrence_requests_checkpoints() -> None:
    """The prefix recurrence must honour the shared request envelope."""
    import inspect

    source = inspect.getsource(ore_operations._compute_admitted_dfinite_prefix)
    assert "request_checkpoint" in source


def test_recurrence_stops_on_a_cancellation_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = differential_series_construct(
        _operator([(0, [(1, 0)]), (1, [(1, 0)])]), _derivatives(1)
    )
    calls = 0
    from jacobian._execution import request_checkpoint as original

    def counting_checkpoint(label: str) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError("cancelled")
        original(label)

    monkeypatch.setattr(ore_operations, "request_checkpoint", counting_checkpoint)
    with pytest.raises(TimeoutError):
        differential_series_generate_prefix(series, 40)
    assert calls > 1


# Negative control: an uncomputed request still returns its exact prefix.
def test_uninterrupted_recurrence_still_computes() -> None:
    series = differential_series_construct(
        _operator([(0, [(1, 0)]), (1, [(1, 0)])]), _derivatives(1)
    )
    assert _prefix(series, 4) == [
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2),
        Fraction(-1, 6),
    ]


# --- the native operation is exported --------------------------------------


def test_native_prefix_operation_is_exported() -> None:
    import jacobian.math.ore_algebras as package
    from jacobian.math.ore_algebras import operations

    assert "differential_series_generate_prefix" in operations.__all__
    assert "differential_series_generate_prefix" in package.__all__
    assert (
        package.differential_series_generate_prefix
        is differential_series_generate_prefix
    )


def test_native_import_matches_the_catalog_operation() -> None:
    from jacobian.math.ore_algebras._tools import TOOLS

    operation = next(
        item
        for item in TOOLS
        if item.operation_id
        == "holonomic.differential_series.generate_finite_prefix.compute"
    )
    request = operation.request_type.model_validate(
        {
            "series": {
                "operator": _operator([(0, [(1, 0)]), (1, [(1, 0)])]).model_dump(),
                "initial_derivatives": _derivatives(1).model_dump(),
                "center": 0,
            },
            "count": 3,
        }
    )
    result = operation.run(request)
    assert [value.as_fraction() for value in result.values] == [
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2),
    ]


# Negative control: an oversized count is still refused.
def test_oversized_prefix_count_is_still_refused() -> None:
    from jacobian.math.number_theory.sequences.core.values import (
        MAX_SEQUENCE_LENGTH,
    )

    series = differential_series_construct(
        _operator([(0, [(1, 0)]), (1, [(1, 0)])]), _derivatives(1)
    )
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ):
        differential_series_generate_prefix(series, MAX_SEQUENCE_LENGTH + 1)
