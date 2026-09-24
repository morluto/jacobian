"""Exact finite action of shift Ore operators on rational sequence prefixes."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras import operations
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.operations import (
    shift_operator_apply_to_sequence_prefix,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(
    numerator: tuple[tuple[Fraction | int, int], ...], denominator=((1, 0),)
) -> dict:
    def _part(terms):
        return {
            "terms": [
                {
                    "coefficient": {
                        "num": Fraction(coefficient).numerator,
                        "den": Fraction(coefficient).denominator,
                    },
                    "exponents": [exponent],
                }
                for coefficient, exponent in sorted(
                    terms, key=lambda term: term[1], reverse=True
                )
            ]
        }

    return {
        "domain": "QQ",
        "variables": ["n"],
        "numerator": _part(numerator),
        "denominator": _part(denominator),
    }


def _operator(terms: tuple[tuple[int, dict], ...]) -> ShiftOreOperator:
    return ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": exponent, "coefficient": coefficient}
                for exponent, coefficient in terms
            ],
        }
    )


def _independent_poly_eval(polynomial, index: int) -> Fraction:
    # Deliberately use the public coefficient payload and direct powers rather
    # than the Ore kernel's private polynomial/evaluation helpers.
    return sum(
        (
            Fraction(term.coefficient.num, term.coefficient.den)
            * index ** term.exponents[0]
            for term in polynomial.terms
        ),
        Fraction(0),
    )


def _independent_rf_eval(function: RationalFunction, index: int) -> Fraction:
    return _independent_poly_eval(function.numerator, index) / _independent_poly_eval(
        function.denominator, index
    )


def test_fibonacci_operator_residual_matches_direct_recurrence_oracle() -> None:
    op = _operator(
        (
            (0, _rf(((-1, 0),))),
            (1, _rf(((-1, 0),))),
            (2, _rf(((1, 0),))),
        )
    )
    source = FiniteRationalSequence(values=(0, 1, 1, 2, 3, 5, 8))

    result = shift_operator_apply_to_sequence_prefix(op, 10, source)

    assert result.start_index == 10
    assert result.sequence == source
    assert [row.index for row in result.residuals] == [10, 11, 12, 13, 14]
    assert result.right_boundary_indices == (15, 16)
    for offset, row in enumerate(result.residuals):
        direct = (
            source.values[offset + 2].as_fraction()
            - source.values[offset + 1].as_fraction()
            - source.values[offset].as_fraction()
        )
        assert row.residual.as_fraction() == direct == 0
        assert [entry.exponent for entry in row.contributions] == [0, 1, 2]
        assert (
            sum(entry.value.as_fraction() for entry in row.contributions)
            == row.residual.as_fraction()
        )
        assert [entry.sequence_index for entry in row.contributions] == [
            row.index + exponent for exponent in (0, 1, 2)
        ]
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_variable_coefficient_residual_uses_direct_rational_evaluation() -> None:
    op = _operator(
        (
            (0, _rf(((-1, 1),))),
            (1, _rf(((1, 0), (1, 1)))),
        )
    )
    source = FiniteRationalSequence(values=(2, 5, -1, 7, 9))
    result = shift_operator_apply_to_sequence_prefix(op, -2, source)

    for offset, row in enumerate(result.residuals):
        index = -2 + offset
        expected = sum(
            (
                _independent_rf_eval(term.coefficient, index)
                * source.values[offset + term.exponent].as_fraction()
                for term in op.terms
            ),
            Fraction(0),
        )
        assert row.residual.as_fraction() == expected
        for contribution, term in zip(row.contributions, op.terms, strict=True):
            assert contribution.coefficient.as_fraction() == _independent_rf_eval(
                term.coefficient, index
            )


def test_rational_coefficient_and_sequence_values_remain_exact() -> None:
    op = _operator(((0, _rf(((Fraction(1, 2), 0),))),))
    source = FiniteRationalSequence(
        values=(
            CanonicalRational.from_fraction(Fraction(2, 3)),
            CanonicalRational.from_fraction(Fraction(-4, 5)),
        )
    )

    result = shift_operator_apply_to_sequence_prefix(op, 7, source)

    assert [row.residual.as_fraction() for row in result.residuals] == [
        Fraction(1, 3),
        Fraction(-2, 5),
    ]


def test_coefficient_pole_excludes_whole_index_and_tail_is_explicit() -> None:
    op = _operator(
        (
            (0, _rf(((1, 0),), ((1, 1), (-1, 0)))),
            (1, _rf(((1, 0),))),
        )
    )
    result = shift_operator_apply_to_sequence_prefix(
        op, 0, FiniteRationalSequence(values=(3, 4, 5, 6))
    )

    assert [row.index for row in result.residuals] == [0, 2]
    assert [(item.index, item.exponents) for item in result.coefficient_poles] == [
        (1, (0,))
    ]
    assert result.right_boundary_indices == (3,)


def test_zero_operator_covers_each_stored_index_without_boundary_loss() -> None:
    result = shift_operator_apply_to_sequence_prefix(
        _operator(()), 4, FiniteRationalSequence(values=(2, 3, 5))
    )
    assert [row.index for row in result.residuals] == [4, 5, 6]
    assert [row.residual.as_fraction() for row in result.residuals] == [0, 0, 0]
    assert result.right_boundary_indices == ()


def test_work_admission_precedes_coefficient_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(operations, "MAX_SHIFT_PREFIX_EVALUATION_CELLS", 0)

    def fail(*args, **kwargs):
        raise AssertionError("coefficient evaluation must follow admission")

    monkeypatch.setattr(operations, "_evaluate_rational_polynomial_at", fail)
    with pytest.raises(OperationResourceAdmissionError, match="evaluation budget"):
        shift_operator_apply_to_sequence_prefix(
            _operator(((0, _rf(((1, 1),))),)),
            0,
            FiniteRationalSequence(values=(1, 2)),
        )


def test_start_index_is_explicit_and_bounded() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="integer-index envelope"):
        shift_operator_apply_to_sequence_prefix(
            _operator(()), 100_001, FiniteRationalSequence(values=(1,))
        )


def test_output_admission_precedes_coefficient_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(operations, "MAX_SHIFT_PREFIX_OUTPUT_BYTES", 0)

    def fail(*args, **kwargs):
        raise AssertionError("coefficient evaluation must follow output admission")

    monkeypatch.setattr(operations, "_evaluate_rational_polynomial_at", fail)
    with pytest.raises(OperationResourceAdmissionError, match="byte budget"):
        shift_operator_apply_to_sequence_prefix(
            _operator(((0, _rf(((1, 0),))),)),
            0,
            FiniteRationalSequence(values=(1,)),
        )
