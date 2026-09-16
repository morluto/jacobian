"""Exact and contract tests for quadratic-form coefficient matrices."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general import (
    coefficient_matrix,
    coefficient_matrix_entries,
    evaluate_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general._models import (
    MAX_COEFFICIENT_MATRIX_AXIS,
    CoefficientMatrixRequest,
    CoefficientMatrixResult,
    EvaluationRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import (
    compute_coefficient_matrix,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalCoordinateVector,
    RationalQuadraticForm,
)


def _rational(numerator: int, denominator: int = 1) -> dict[str, int]:
    return {"num": numerator, "den": denominator}


def _binary_form() -> dict[str, object]:
    return {
        "axis": ["x", "y"],
        "diagonal_coefficients": [_rational(2), _rational(5)],
        "cross_terms": [
            {"left": 0, "right": 1, "coefficient": _rational(3)},
        ],
    }


def test_known_answer_with_odd_cross_term_halved() -> None:
    result = compute_coefficient_matrix(
        CoefficientMatrixRequest.model_validate({"form": _binary_form()})
    )

    assert result.convention == "Q_X_EQUALS_X_TRANSPOSE_A_X"
    assert result.matrix.row_count == 2
    assert result.matrix.column_count == 2
    assert result.matrix.entries == (
        (CanonicalRational(num=2, den=1), CanonicalRational(num=3, den=2)),
        (CanonicalRational(num=3, den=2), CanonicalRational(num=5, den=1)),
    )


def test_native_entries_are_exact_fractions() -> None:
    form = RationalQuadraticForm.model_validate(_binary_form())

    assert coefficient_matrix_entries(form) == (
        (Fraction(2), Fraction(3, 2)),
        (Fraction(3, 2), Fraction(5)),
    )


def test_zero_dimensional_form_yields_empty_matrix() -> None:
    result = compute_coefficient_matrix(
        CoefficientMatrixRequest.model_validate(
            {"form": {"axis": [], "diagonal_coefficients": []}}
        )
    )

    assert result.matrix.row_count == 0
    assert result.matrix.column_count == 0
    assert result.matrix.entries == ()


def test_diagonal_form_yields_diagonal_matrix() -> None:
    result = compute_coefficient_matrix(
        CoefficientMatrixRequest.model_validate(
            {
                "form": {
                    "axis": ["u", "v", "w"],
                    "diagonal_coefficients": [
                        _rational(1),
                        _rational(-2, 3),
                        _rational(0),
                    ],
                }
            }
        )
    )

    assert result.matrix.entries == (
        (
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
        ),
        (
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=-2, den=3),
            CanonicalRational(num=0, den=1),
        ),
        (
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )


def _replay_quadratic(
    entries: tuple[tuple[Fraction, ...], ...], point: tuple[Fraction, ...]
) -> Fraction:
    return sum(
        entries[i][j] * point[i] * point[j]
        for i in range(len(point))
        for j in range(len(point))
    )


@pytest.mark.parametrize(
    "point",
    [
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(2)),
        (Fraction(-3, 7), Fraction(5, 11)),
        (Fraction(4), Fraction(-1, 3)),
    ],
)
def test_defining_invariant_matrix_replays_evaluation(
    point: tuple[Fraction, Fraction],
) -> None:
    form = RationalQuadraticForm.model_validate(_binary_form())
    vector = RationalCoordinateVector(
        axis=form.axis,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in point),
    )

    assert _replay_quadratic(coefficient_matrix_entries(form), point) == (
        evaluate_rational_quadratic_form(form, vector)
    )


def test_symmetric_replay_on_ternary_form() -> None:
    form = RationalQuadraticForm.model_validate(
        {
            "axis": ["x", "y", "z"],
            "diagonal_coefficients": [_rational(1), _rational(2), _rational(3)],
            "cross_terms": [
                {"left": 0, "right": 1, "coefficient": _rational(1)},
                {"left": 0, "right": 2, "coefficient": _rational(-5, 2)},
                {"left": 1, "right": 2, "coefficient": _rational(4, 3)},
            ],
        }
    )
    entries = coefficient_matrix_entries(form)

    for i in range(3):
        for j in range(3):
            assert entries[i][j] == entries[j][i]
    assert entries[0][1] == Fraction(1, 2)
    assert entries[0][2] == Fraction(-5, 4)
    assert entries[1][2] == Fraction(2, 3)
    point = (Fraction(1, 3), Fraction(-2), Fraction(7, 5))
    vector = RationalCoordinateVector(
        axis=form.axis,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in point),
    )
    assert _replay_quadratic(entries, point) == evaluate_rational_quadratic_form(
        form, vector
    )


def test_native_and_catalog_results_agree() -> None:
    form = RationalQuadraticForm.model_validate(_binary_form())
    native = coefficient_matrix(form)
    catalog = compute_coefficient_matrix(CoefficientMatrixRequest(form=form))

    assert catalog.form == form
    assert catalog.matrix == native
    assert (
        CoefficientMatrixResult.model_validate_json(catalog.model_dump_json())
        == catalog
    )


def test_forged_shape_mismatch_is_rejected() -> None:
    form = RationalQuadraticForm.model_validate(_binary_form())
    native = coefficient_matrix(form)

    with pytest.raises(ValidationError):
        CoefficientMatrixResult(
            form=form,
            matrix=native.model_copy(update={"row_count": 3, "column_count": 3}),
        )


def test_dimension_above_envelope_is_refused_before_arithmetic() -> None:
    dimension = MAX_COEFFICIENT_MATRIX_AXIS + 1
    form = RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(dimension)),
        diagonal_coefficients=tuple(
            CanonicalRational(num=0, den=1) for _ in range(dimension)
        ),
    )

    with pytest.raises(OperationResourceAdmissionError):
        coefficient_matrix(form)
    with pytest.raises(OperationResourceAdmissionError):
        compute_coefficient_matrix(CoefficientMatrixRequest(form=form))


def test_dimension_at_envelope_is_accepted() -> None:
    dimension = MAX_COEFFICIENT_MATRIX_AXIS
    form = RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(dimension)),
        diagonal_coefficients=tuple(
            CanonicalRational(num=1 if i == 0 else 0, den=1) for i in range(dimension)
        ),
    )

    result = compute_coefficient_matrix(CoefficientMatrixRequest(form=form))
    assert result.matrix.row_count == dimension
    assert result.matrix.entries[0][0] == CanonicalRational(num=1, den=1)


def test_native_rejects_a_non_form_value() -> None:
    with pytest.raises(OperationDomainValidationError):
        coefficient_matrix_entries("not-a-form")  # type: ignore[arg-type]


def test_evaluation_request_still_requires_shared_axis() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(
            {
                "form": _binary_form(),
                "vector": {"axis": ["x"], "coordinates": [_rational(1)]},
            }
        )
