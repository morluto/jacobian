"""Regressions for the Ore division and recurrence-to-OGF boundaries.

Each repaired case is paired with a negative control showing the enclosing
bound or domain rejection is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    DifferentialOreOperator,
    DifferentialOreTerm,
)
from jacobian.math.ore_algebras.differential_left_division.operations import (
    differential_operator_left_divide_monic,
)
from jacobian.math.ore_algebras.differential_right_division.operations import (
    differential_operator_right_divide_monic,
)
from jacobian.math.ore_algebras.recurrence_to_ogf.operations import (
    polynomial_recurrence_to_ogf_equation,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _one() -> CanonicalRational:
    return CanonicalRational(num=1, den=1)


def _unit_denominator() -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=(RationalPolynomialTerm(exponents=(0,), coefficient=_one()),)
    )


def _rf(terms: Sequence[tuple[int, int]]) -> RationalFunction:
    """``terms`` are ``(exponent, coefficient)`` pairs in the single variable."""
    return RationalFunction(
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=tuple(
                sorted(
                    (
                        RationalPolynomialTerm(
                            exponents=(exponent,),
                            coefficient=CanonicalRational(num=coefficient, den=1),
                        )
                        for exponent, coefficient in terms
                    ),
                    key=lambda term: term.exponents,
                    reverse=True,
                )
            )
        ),
        denominator=_unit_denominator(),
    )


def _operator(
    terms: Sequence[tuple[int, int, int]],
) -> DifferentialOreOperator:
    """``terms`` are ``(differential order, x degree, coefficient)`` triples."""
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": order,
                    "coefficient": _rf([(degree, coefficient)]).model_dump(),
                }
                for order, degree, coefficient in sorted(terms, key=lambda t: t[0])
            ],
        }
    )


def _monic(order: int, degree: int = 0) -> DifferentialOreOperator:
    return _operator([(order, degree, 1)])


# --- left division admits forced reductions before the input ceilings -------


def test_left_forced_identity_beyond_the_order_ceiling() -> None:
    """A = B = D^5 reduces immediately to Q=1, R=0."""
    dividend = _monic(5)
    result = differential_operator_left_divide_monic(dividend, dividend)
    assert result.quotient.order == 0
    assert result.remainder.order == -1


def test_left_forced_zero_quotient_beyond_the_order_ceiling() -> None:
    """A = 1 with B = D^5 cancels nothing, so Q=0 and R=1."""
    result = differential_operator_left_divide_monic(_monic(0), _monic(5))
    assert result.quotient.order == -1
    assert result.remainder.order == 0


def test_left_forced_identity_beyond_the_degree_ceiling() -> None:
    """A = B = D + x^3 exceeds the degree-2 ceiling but cancels exactly."""
    dividend = _operator([(0, 3, 1), (1, 0, 1)])
    result = differential_operator_left_divide_monic(dividend, dividend)
    assert result.quotient.order == 0
    assert result.remainder.order == -1


def test_left_forced_zero_quotient_beyond_the_degree_ceiling() -> None:
    result = differential_operator_left_divide_monic(
        _operator([(0, 3, 1)]), _operator([(0, 3, 1), (1, 0, 1)])
    )
    assert result.quotient.order == -1
    assert result.remainder.order == 0


# Negative control: a genuine reduction beyond the ceilings is still refused,
# because the division loop would really perform the quotient growth.
def test_left_general_division_still_enforces_the_order_ceiling() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        differential_operator_left_divide_monic(
            _operator([(0, 0, 1), (6, 0, 1)]), _monic(5)
        )
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.differential_left_division_order"
    )


def test_left_general_division_still_enforces_the_degree_ceiling() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        differential_operator_left_divide_monic(
            _operator([(0, 3, 1), (2, 0, 1), (3, 0, 1)]), _monic(2)
        )
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.differential_left_division_coefficient_bound"
    )


def test_left_forced_result_still_needs_a_monic_divisor() -> None:
    """The forced shortcut does not excuse a non-monic divisor."""
    with pytest.raises(OperationDomainValidationError) as error:
        differential_operator_left_divide_monic(
            _operator([(1, 0, 2)]), _operator([(1, 0, 2)])
        )
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.differential_left_division_monic"
    )


# --- the entry points canonicalize and admit once ---------------------------


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("mapping", (False, True))
def test_entry_point_validates_and_admits_once(
    side: str, mapping: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jacobian.math.ore_algebras.differential_left_division import operations as left
    from jacobian.math.ore_algebras.differential_left_division._models import (
        DifferentialLeftDivisionRequest,
    )
    from jacobian.math.ore_algebras.differential_right_division import (
        operations as right,
    )
    from jacobian.math.ore_algebras.differential_right_division._models import (
        DifferentialRightDivisionRequest,
    )

    operations = left if side == "left" else right
    request_type = (
        DifferentialLeftDivisionRequest
        if side == "left"
        else DifferentialRightDivisionRequest
    )
    divide = (
        differential_operator_left_divide_monic
        if side == "left"
        else differential_operator_right_divide_monic
    )
    # D(D+x) = D^2+xD+1, while (D+x)D = D^2+xD. The two
    # nontrivial divisions have the same quotient and different remainders.
    dividend = _operator([(0, 0, 1), (1, 1, 1), (2, 0, 1)])
    divisor = _operator([(0, 1, 1), (1, 0, 1)])
    expected_quotient = _monic(1)
    expected_remainder = _monic(0) if side == "left" else _operator([])
    operands = (dividend.model_dump(), divisor.model_dump())
    request_calls: list[object] = []
    operand_calls: list[object] = []
    admission_calls: list[tuple[DifferentialOreOperator, DifferentialOreOperator]] = []
    validate_request = request_type.model_validate
    validate_operator = DifferentialOreOperator.model_validate
    admit: Callable[
        [DifferentialOreOperator, DifferentialOreOperator], tuple[int, int, int]
    ] = operations._admit

    def observed_request(
        cls: type[StrictModel], value: object, *args: Any, **kwargs: Any
    ) -> StrictModel:
        request_calls.append(value)
        return validate_request(value, *args, **kwargs)

    def observed_operator(
        cls: type[DifferentialOreOperator], value: object, *args: Any, **kwargs: Any
    ) -> DifferentialOreOperator:
        if value in operands:
            operand_calls.append(value)
        return validate_operator(value, *args, **kwargs)

    def observed_admission(
        a: DifferentialOreOperator, b: DifferentialOreOperator
    ) -> tuple[int, int, int]:
        admission_calls.append((a, b))
        return admit(a, b)

    monkeypatch.setattr(request_type, "model_validate", classmethod(observed_request))
    monkeypatch.setattr(
        DifferentialOreOperator, "model_validate", classmethod(observed_operator)
    )
    monkeypatch.setattr(operations, "_admit", observed_admission)
    result = divide(
        operands[0] if mapping else dividend,
        operands[1] if mapping else divisor,
    )
    assert result.quotient == expected_quotient
    assert result.remainder == expected_remainder
    assert len(request_calls) == 1
    assert operand_calls == list(operands)
    assert admission_calls == [(dividend, divisor)]


def test_single_admission_still_rejects_a_forged_operator() -> None:
    """Structural canonicalization is retained by the cheaper entry point."""
    forged = DifferentialOreOperator.model_construct(
        variable="x",
        terms=(
            DifferentialOreTerm.model_construct(
                order=0,
                coefficient=RationalFunction.model_construct(
                    variables=("x",),
                    numerator=SparseRationalPolynomial.model_construct(
                        terms=(
                            RationalPolynomialTerm.model_construct(
                                exponents=(0,), coefficient=_one()
                            ),
                        )
                    ),
                    denominator=_unit_denominator(),
                ),
            ),
            DifferentialOreTerm.model_construct(
                order=0,
                coefficient=_rf([(0, 1)]),
            ),
        ),
    )
    # Duplicate differential orders are not canonical, so one admission is
    # enough to refuse the value.
    with pytest.raises(OperationDomainValidationError):
        differential_operator_left_divide_monic(forged, _monic(1))


# --- right division preserves its forced identity past the shift cap -------


def test_right_forced_identity_beyond_the_shift_coefficient_cap() -> None:
    """A = B = D + x^65 has the forced result Q=1, R=0."""
    dividend = _operator([(0, 65, 1), (1, 0, 1)])
    result = differential_operator_right_divide_monic(dividend, dividend)
    assert result.quotient.order == 0
    assert result.remainder.order == -1


# Negative control: a general right division beyond the shared shift budget
# still respects this operation's own output envelope.
def test_right_general_division_still_enforces_its_envelope() -> None:
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ):
        differential_operator_right_divide_monic(
            _operator([(0, 0, 1), (2, 0, 1)]), _operator([(1, 0, 2)])
        )


def test_right_non_polynomial_coefficient_is_still_refused() -> None:
    rational = RationalFunction.model_construct(
        variables=("x",),
        numerator=_rf([(0, 1)]).numerator,
        denominator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(exponents=(1,), coefficient=_one()),)
        ),
    )
    dividend = DifferentialOreOperator.model_construct(
        variable="x",
        terms=(DifferentialOreTerm.model_construct(order=0, coefficient=rational),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        differential_operator_right_divide_monic(dividend, _monic(1))
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.differential_right_division_polynomial_domain"
    )


# --- the OGF conversion does not charge shift growth twice -----------------


def _ogf_rf(terms: Sequence[tuple[int, int]]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "variables": ["n"],
        "numerator": {
            "terms": [
                {"coefficient": {"num": value, "den": 1}, "exponents": [degree]}
                for degree, value in sorted(terms, reverse=True)
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }


def test_order_sixteen_recurrence_with_one_wide_forcing_is_admitted() -> None:
    """Only q_16(n)=n^16 with a_15 = 10^110 yields an 111-digit forcing."""
    recurrence = {
        "variable": "n",
        "terms": [{"exponent": 16, "coefficient": _ogf_rf([(16, 1)])}],
    }
    initial = {"values": [{"num": 0, "den": 1}] * 15 + [{"num": 10**110, "den": 1}]}
    result = polynomial_recurrence_to_ogf_equation(recurrence, initial)
    forcing = result.forcing.numerator.terms
    assert len(forcing) == 1
    numerator = abs(forcing[0].coefficient.num)
    assert numerator.bit_length() <= 400


# Negative control: a forcing beyond the exact rational carrier is still
# refused once the genuinely measured growth is applied.
def test_forcing_beyond_the_coefficient_carrier_is_still_refused() -> None:
    recurrence = {
        "variable": "n",
        "terms": [{"exponent": 16, "coefficient": _ogf_rf([(16, 1)])}],
    }
    initial = {"values": [{"num": 0, "den": 1}] * 15 + [{"num": 10**200, "den": 1}]}
    with pytest.raises(OperationResourceAdmissionError) as error:
        polynomial_recurrence_to_ogf_equation(recurrence, initial)
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.recurrence_ogf_coefficient_digits"
    )


def test_ogf_known_answer_still_matches_sympy() -> None:
    """The Fibonacci OGF equation is unchanged by the bound repair."""
    import sympy as sp

    result = polynomial_recurrence_to_ogf_equation(
        {
            "variable": "n",
            "terms": [
                {"exponent": 0, "coefficient": _ogf_rf([(0, -1)])},
                {"exponent": 1, "coefficient": _ogf_rf([(0, -1)])},
                {"exponent": 2, "coefficient": _ogf_rf([(0, 1)])},
            ],
        },
        {"values": [{"num": 1, "den": 1}, {"num": 1, "den": 1}]},
    )
    x = sp.Symbol("x")

    def as_sympy(coefficient: object) -> sp.Expr:
        terms = coefficient.numerator.terms  # type: ignore[attr-defined]
        return sum(
            sp.Rational(
                term.coefficient.as_fraction().numerator,
                term.coefficient.as_fraction().denominator,
            )
            * x ** term.exponents[0]
            for term in terms
        )

    left = sum(
        as_sympy(operator_term.coefficient)
        * sp.diff(1 / (1 - x - x**2), x, operator_term.order)
        for operator_term in result.differential_operator.terms
    )
    assert sp.cancel(left - as_sympy(result.forcing)) == 0
