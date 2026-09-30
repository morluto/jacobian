import json
from fractions import Fraction
from math import factorial
from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    MAX_COEFFICIENT_RECURRENCE_BOUNDARY_INDEX,
    MAX_SHIFT_ORDER,
    DifferentialOreOperator,
    PolynomialRecurrencePrefixRequest,
)
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import (
    differential_operator_to_coefficient_recurrence,
    polynomial_recurrence_generate_prefix,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(terms: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x"],
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


def _operator(*terms: tuple[int, list[tuple[int, int]]]) -> dict[str, Any]:
    return {
        "variable": "x",
        "terms": [
            {"order": order, "coefficient": _rf(polynomial)}
            for order, polynomial in terms
        ],
    }


def _polynomial(coefficient: RationalFunction) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in coefficient.numerator.terms
    }


def _index_value(coefficient: RationalFunction, index: int) -> Fraction:
    """Evaluate one cleared-denominator recurrence coefficient at an index."""
    total = Fraction(0)
    for term in coefficient.numerator.terms:
        total += term.coefficient.as_fraction() * Fraction(index) ** term.exponents[0]
    for term in coefficient.denominator.terms:
        total /= term.coefficient.as_fraction() * Fraction(index) ** term.exponents[0]
    return total


def test_exponential_equation_yields_factorial_recurrence() -> None:
    operator = DifferentialOreOperator.model_validate(
        _operator((0, [(0, -1)]), (1, [(0, 1)]))
    )
    result = differential_operator_to_coefficient_recurrence(operator)
    assert result.valid_from == 0
    assert [
        (term.exponent, _polynomial(term.coefficient))
        for term in result.recurrence.terms
    ] == [
        (0, {0: Fraction(-1)}),
        (1, {0: Fraction(1), 1: Fraction(1)}),
    ]
    assert result.boundary_rows == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result
    # Exact independent solution a_n=1/n! satisfies every emitted row.
    for n in range(result.valid_from, 24):
        lhs = sum(
            sum(
                value * n**degree
                for degree, value in _polynomial(term.coefficient).items()
            )
            / factorial(n + term.exponent)
            for term in result.recurrence.terms
        )
        assert lhs == 0
    prefix = polynomial_recurrence_generate_prefix(
        result.recurrence,
        result.valid_from,
        FiniteRationalSequence.model_validate({"values": [1]}),
        5,
    )
    assert [value.as_fraction() for value in prefix.values.values] == [
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 6),
        Fraction(1, 24),
        Fraction(1, 120),
    ]


def test_sinh_equation_has_exact_second_order_recurrence() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (2, [(0, 1)]))
    )
    recurrence = {
        term.exponent: _polynomial(term.coefficient) for term in result.recurrence.terms
    }
    assert recurrence == {
        0: {0: Fraction(-1)},
        2: {0: Fraction(2), 1: Fraction(3), 2: Fraction(1)},
    }
    # The independent exact solution y=e^x has coefficients a_n=1/n!.
    coefficients = [Fraction(1, factorial(index)) for index in range(40)]
    for n in range(18):
        lhs = -coefficients[n] + (n + 2) * (n + 1) * coefficients[n + 2]
        assert lhs == 0


def test_low_degree_boundary_row_is_retained() -> None:
    # x*y' - y = 0 has the exceptional row -a_0=0 before (n-1)a_n=0. The
    # leading coefficient n-1 vanishes at n=1, so the start advances to the
    # first index where the equation can actually solve for a_n, and the
    # skipped index 1 is retained as a boundary row.
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (1, [(1, 1)]))
    )
    assert result.valid_from == 2
    assert [row.degree for row in result.boundary_rows] == [0, 1]
    assert [
        (term.index, term.coefficient.as_fraction())
        for term in result.boundary_rows[0].terms
    ] == [(0, Fraction(-1))]
    leading = max(result.recurrence.terms, key=lambda term: term.exponent)
    assert _index_value(leading.coefficient, 1) == 0
    assert _index_value(leading.coefficient, result.valid_from) != 0
    assert result.recurrence.terms[0].exponent == 0
    assert _polynomial(result.recurrence.terms[0].coefficient) == {
        0: Fraction(-1),
        1: Fraction(1),
    }


def test_rational_function_ode_coefficients_are_rejected() -> None:
    rational = _rf([(0, 1)])
    rational["denominator"] = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        differential_operator_to_coefficient_recurrence(
            {"variable": "x", "terms": [{"order": 0, "coefficient": rational}]}
        )


def test_catalog_example_dispatches_the_exact_recurrence_value() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "holonomic.differential_operator.to_coefficient_recurrence.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.valid_from == 0
    assert [term.exponent for term in result.recurrence.terms] == [0, 1]


def test_multiple_mixed_boundary_rows_match_direct_coefficient_extraction() -> None:
    # L = 1 + x*D + x^2*D^2 gives a_0=0, 2*a_1=0, then
    # (n^2+1)*a_n=0 for n >= 2.
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, 1)]), (1, [(1, 1)]), (2, [(2, 1)]))
    )
    assert result.valid_from == 2
    assert [
        (
            row.degree,
            [(term.index, term.coefficient.as_fraction()) for term in row.terms],
        )
        for row in result.boundary_rows
    ] == [
        (0, [(0, Fraction(1))]),
        (1, [(1, Fraction(2))]),
    ]
    stable = result.recurrence.terms
    assert len(stable) == 1
    assert stable[0].exponent == 0
    assert _polynomial(stable[0].coefficient) == {0: Fraction(1), 2: Fraction(1)}

    coefficients = [Fraction(2), Fraction(-3), Fraction(5), Fraction(7), Fraction(11)]
    for row in result.boundary_rows:
        lhs = sum(
            term.coefficient.as_fraction() * coefficients[term.index]
            for term in row.terms
        )
        m = row.degree
        direct = (
            coefficients[m]
            + (m * coefficients[m] if m >= 1 else 0)
            + (m * (m - 1) * coefficients[m] if m >= 2 else 0)
        )
        assert lhs == direct
    n = 2
    lhs = sum(
        sum(
            coefficient * n**power
            for power, coefficient in _polynomial(term.coefficient).items()
        )
        * coefficients[n + term.exponent]
        for term in stable
    )
    direct = coefficients[n] + n * coefficients[n] + n * (n - 1) * coefficients[n]
    assert lhs == direct


def test_boundary_work_is_admitted_before_any_boundary_expansion() -> None:
    """The work bound is enforced, so it has to be reachable.

    This input keeps the shift span at the carrier's limit while the expansion
    work still exceeds the budget. Realistic ODEs span only a few, so an input
    built from a wide coefficient range is rejected by the shift check first
    and never reaches this bound.
    """

    operator = _operator(
        *((order, [(order + step, 1) for step in range(17)]) for order in range(16))
    )
    with pytest.raises(OperationResourceAdmissionError, match="work budget"):
        differential_operator_to_coefficient_recurrence(operator)


def test_transform_result_composes_unchanged_with_prefix_generation() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (1, [(0, 1)]))
    )
    request = {
        "operator": result.recurrence.model_dump(),
        "start_index": result.valid_from,
        "initial_values": {"values": [1]},
        "steps": 5,
    }
    prefix_request = PolynomialRecurrencePrefixRequest.model_validate(request)
    assert [
        value.as_fraction()
        for value in polynomial_recurrence_generate_prefix(
            prefix_request.operator,
            prefix_request.start_index,
            prefix_request.initial_values,
            prefix_request.steps,
        ).values.values
    ] == [
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 6),
        Fraction(1, 24),
        Fraction(1, 120),
    ]


def test_generated_coefficients_stay_within_downstream_shift_envelope() -> None:
    # Although each input coefficient fits 64 digits, 10^63*(n+16)_16 has
    # a 78-digit constant coefficient and cannot be consumed as a shift op.
    operator = _operator((0, [(0, 1)]), (16, [(0, 10**63)]))
    with pytest.raises(
        OperationResourceAdmissionError, match="coefficient recurrence coefficient"
    ):
        differential_operator_to_coefficient_recurrence(operator)


def test_output_uses_canonical_shift_operator_through_maximum_shift_span() -> None:
    """The result is a canonical shift operator at its widest reachable span.

    The order of the returned shift operator is the full shift span, which the
    shared ``ShiftOreOperator`` envelope caps at ``MAX_SHIFT_ORDER``. The
    recurrence consumer inherits that cap, so a wider recurrence could not be
    generated usefully. This case sits exactly at the limit.
    """

    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(8, 1)]), (8, [(0, 1)]))
    )
    assert result.recurrence.order == MAX_SHIFT_ORDER
    assert (
        type(result.recurrence).model_validate_json(result.recurrence.model_dump_json())
        == result.recurrence
    )


def test_valid_from_advances_past_a_singular_leading_coefficient() -> None:
    """`valid_from` must be an index the leading coefficient is nonzero at.

    For `L = 1 - D + x D^2` the leading coefficient is `n^2 - 1`, so the
    largest-coefficient-degree start of 1 is singular. The documented consumer
    rejects a singular start immediately, so handing it this result unchanged
    did not compose. The start now advances to the first admissible index and
    the skipped equation is retained as a boundary row, since it cannot solve
    for a coefficient.
    """
    operator = DifferentialOreOperator.model_validate(
        _operator((0, [(0, 1)]), (1, [(0, -1)]), (2, [(1, 1)]))
    )
    result = differential_operator_to_coefficient_recurrence(operator)

    leading = max(result.recurrence.terms, key=lambda term: term.exponent)
    assert leading.exponent == result.recurrence.order
    assert _polynomial(leading.coefficient) == {2: Fraction(1), 0: Fraction(-1)}
    assert _index_value(leading.coefficient, 1) == 0
    assert result.valid_from == 2
    assert _index_value(leading.coefficient, result.valid_from) == 3
    # the skipped index is retained rather than dropped
    assert [row.degree for row in result.boundary_rows] == [0, 1]


def test_generated_recurrence_term_count_is_admitted() -> None:
    """A generated recurrence must fit its own carrier before construction.

    A single order whose coefficient is `1 + x + ... + x^16` generates all 17
    shifts from 0 through 16. `ShiftOreOperator` permits 16 terms, so the
    result construction used to raise an unclassified Pydantic error after
    admission had already accepted the request.
    """
    operator = DifferentialOreOperator.model_validate(
        _operator((0, [(degree, 1) for degree in range(17)]))
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        differential_operator_to_coefficient_recurrence(operator)
    assert error.value.errors()[0]["type"] == (
        "ore_algebra.coefficient_recurrence_terms"
    )

    # 16 generated shifts still fit and are admitted
    fitting = DifferentialOreOperator.model_validate(
        _operator((0, [(degree, 1) for degree in range(16)]))
    )
    assert (
        len(differential_operator_to_coefficient_recurrence(fitting).recurrence.terms)
        == 16
    )


def test_positive_coordinate_shift_does_not_invent_boundary_rows() -> None:
    # D^4 + D^16 has n=m+4. The recurrence starts at n=4, already covering
    # Taylor degree m=0; there are no exceptional Taylor equations.
    result = differential_operator_to_coefficient_recurrence(
        _operator((4, [(0, 1)]), (16, [(0, 1)]))
    )
    assert result.valid_from == 4
    assert result.boundary_rows == ()


def test_negative_coordinate_shift_retains_initial_taylor_constraints() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, 1), (8, 1)]))
    )
    assert result.valid_from == 0
    assert [row.degree for row in result.boundary_rows] == list(range(8))
    assert [
        [(term.index, term.coefficient.as_fraction()) for term in row.terms]
        for row in result.boundary_rows
    ] == [[(index, Fraction(1))] for index in range(8)]


def test_regular_start_passes_all_future_integral_singularities() -> None:
    # L=(8+x)-5xD+x^2D^2 has normalized leading coefficient
    # (n-1)(n-3). n=2 is regular but is not a valid stable start.
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, 8), (1, 1)]), (1, [(1, -5)]), (2, [(2, 1)]))
    )
    assert result.valid_from == 4
    assert [row.degree for row in result.boundary_rows] == list(range(5))
    decoded = type(result).model_validate_json(result.model_dump_json())
    prefix = polynomial_recurrence_generate_prefix(
        decoded.recurrence,
        decoded.valid_from,
        FiniteRationalSequence.model_validate({"values": [7]}),
        8,
    )
    values = [value.as_fraction() for value in prefix.values.values]
    for offset in range(8):
        n = decoded.valid_from + offset
        assert values[offset] + (n - 1) * (n - 3) * values[offset + 1] == 0
    # Independent coefficient extraction at Taylor degree m, including the
    # equations skipped at both integral roots.
    coefficients = [Fraction(index + 1) for index in range(6)]
    for row in result.boundary_rows:
        m = row.degree
        observed = sum(
            term.coefficient.as_fraction() * coefficients[term.index]
            for term in row.terms
        )
        expected = (8 - 5 * m + m * (m - 1)) * coefficients[m]
        if m:
            expected += coefficients[m - 1]
        assert observed == expected


@pytest.mark.parametrize("root", [16, 17])
def test_advanced_start_and_boundary_indices_exceed_shift_order(root: int) -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((1, [(0, -root)]), (2, [(1, 1)]), (16, [(16, 1)]))
    )
    assert result.valid_from == root + 1
    assert [row.degree for row in result.boundary_rows] == list(range(root + 1))
    assert max(term.index for row in result.boundary_rows for term in row.terms) == root
    assert (
        max(term.index for row in result.boundary_rows for term in row.terms)
        <= MAX_COEFFICIENT_RECURRENCE_BOUNDARY_INDEX
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    prefix = polynomial_recurrence_generate_prefix(
        result.recurrence,
        result.valid_from,
        FiniteRationalSequence.model_validate({"values": [1]}),
        3,
    )
    values = [value.as_fraction() for value in prefix.values.values]
    for offset in range(3):
        n = result.valid_from + offset
        assert (
            factorial(n) // factorial(n - 16) * values[offset]
            + (n + 1) * (n - root) * values[offset + 1]
            == 0
        )


@pytest.mark.parametrize("root", [10**50])
def test_far_singularities_refuse_before_boundary_materialization(
    monkeypatch: pytest.MonkeyPatch, root: int
) -> None:
    import jacobian.math.ore_algebras.operations as operations

    def fail_if_expanded(*_args: object, **_kwargs: object) -> None:
        pytest.fail("oversized boundary rows must be refused before construction")

    monkeypatch.setattr(
        operations, "_coefficient_recurrence_boundary_rows", fail_if_expanded
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        differential_operator_to_coefficient_recurrence(
            _operator((0, [(0, -root)]), (1, [(1, 1)]))
        )
    assert (
        error.value.errors()[0]["type"]
        == "ore_algebra.coefficient_recurrence_boundary_rows"
    )


def test_boundary_row_limit_remains_accepted() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -63)]), (1, [(1, 1)]))
    )
    assert result.valid_from == 64
    assert [row.degree for row in result.boundary_rows] == list(range(64))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_nonintegral_future_root_does_not_delay_the_start() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -131)]), (1, [(1, 2)]))
    )
    assert result.valid_from == 1
    assert len(result.boundary_rows) == 1
