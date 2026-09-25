from __future__ import annotations

from itertools import permutations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import (
    MatrixAssignmentRequest,
    MatrixFinitePowerSumRequest,
)
from jacobian.math.polynomials.tropical._tools import compute_finite_power_sum
from jacobian.math.polynomials.tropical.operations import (
    tropical_assignment_profile,
    tropical_matrix_power,
    tropical_polynomial_add,
    tropical_polynomial_evaluate,
    tropical_polynomial_multiply,
    tropical_scalar_multiply,
    tropical_vector_projectivize,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_SCALAR_DIGITS,
    TropicalMatrix,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
)


def _semiring() -> TropicalSemiring:
    return TropicalSemiring(convention="MIN_PLUS", base="ZZ")


def _scalar(value: CanonicalRational) -> TropicalScalar:
    return TropicalScalar(semiring=_semiring(), kind="FINITE", value=value)


def _matrix() -> TropicalMatrix:
    semiring = _semiring()
    return TropicalMatrix(
        semiring=semiring,
        row_axis=("a", "b"),
        column_axis=("a", "b"),
        entries=(
            (
                _scalar(CanonicalRational.from_integer_ratio(0, 1)),
                _scalar(CanonicalRational.from_integer_ratio(1, 1)),
            ),
            (
                _scalar(CanonicalRational.from_integer_ratio(2, 1)),
                _scalar(CanonicalRational.from_integer_ratio(3, 1)),
            ),
        ),
    )


def test_tropical_scalar_output_growth_is_admitted_before_construction() -> None:
    value = 10**MAX_TROPICAL_SCALAR_DIGITS - 1
    left = _scalar(CanonicalRational.from_integer_ratio(value, 1))
    right = _scalar(CanonicalRational.from_integer_ratio(value, 1))
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_scalar_multiply(left, right)
    assert error.value.errors()[0]["type"] == "tropical.scalar_output_budget"


def test_finite_power_sum_retains_source_and_cutoff_after_wire_round_trip() -> None:
    source = _matrix()
    result = compute_finite_power_sum(
        MatrixFinitePowerSumRequest(matrix=source, max_power=2)
    )
    assert result.source_matrix == source
    assert result.max_power == 2
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.source_matrix == source
    assert restored.matrix == result.matrix
    assert restored.winning_lengths == result.winning_lengths


def test_finite_power_sum_identity_is_complete_at_zero() -> None:
    source = _matrix()
    result = compute_finite_power_sum(
        MatrixFinitePowerSumRequest(matrix=source, max_power=0)
    )
    assert result.matrix.entries[0][0].value == CanonicalRational.from_integer_ratio(
        0, 1
    )
    assert result.matrix.entries[1][1].value == CanonicalRational.from_integer_ratio(
        0, 1
    )
    assert result.matrix.entries[0][1].kind == "POSITIVE_INFINITY"
    assert result.matrix.entries[1][0].kind == "POSITIVE_INFINITY"


def test_polynomial_add_and_evaluate_admit_untouched_coefficients() -> None:
    semiring = _semiring()
    huge = TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(10**9_000, 1),
    )
    polynomial = TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=(TropicalPolynomialTerm(exponents=(0,), coefficient=huge),),
    )
    empty = TropicalPolynomial(semiring=semiring, variables=("x",), terms=())
    point = TropicalVector(
        semiring=semiring,
        axis=("x",),
        entries=(_scalar(CanonicalRational.from_integer_ratio(0, 1)),),
    )
    with pytest.raises(OperationResourceAdmissionError):
        tropical_polynomial_add(polynomial, empty)
    with pytest.raises(OperationResourceAdmissionError):
        tropical_polynomial_evaluate(polynomial, point)


def test_polynomial_product_admits_exponent_output_before_model_construction() -> None:
    semiring = _semiring()
    left = TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=(
            TropicalPolynomialTerm(
                exponents=(1024,),
                coefficient=_scalar(CanonicalRational.from_integer_ratio(1, 1)),
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        tropical_polynomial_multiply(left, left)


def test_assignment_profile_keeps_all_ties_at_infinity() -> None:
    semiring = _semiring()
    infinity = TropicalScalar(semiring=semiring, kind="POSITIVE_INFINITY", value=None)
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=("a", "b", "c"),
        column_axis=("a", "b", "c"),
        entries=((infinity,) * 3,) * 3,
    )
    value, witnesses = tropical_assignment_profile(matrix)
    assert value.kind == "POSITIVE_INFINITY"
    assert len(witnesses) == 6
    assert set(witnesses) == {
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    }


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_assignment_profile_matches_distinct_row_and_column_axes(
    convention: str,
) -> None:
    semiring = TropicalSemiring(convention=convention, base="ZZ")  # type: ignore[arg-type]
    values = ((1, 4, 8), (7, 2, 6), (5, 9, 3))
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=("worker-a", "worker-b", "worker-c"),
        column_axis=("task-x", "task-y", "task-z"),
        entries=tuple(
            tuple(
                TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(value, 1),
                )
                for value in row
            )
            for row in values
        ),
    )

    request = MatrixAssignmentRequest(matrix=matrix)
    assert request.matrix == matrix

    optimum, assignments = tropical_assignment_profile(matrix)

    scored = {
        permutation: sum(values[row][column] for row, column in enumerate(permutation))
        for permutation in permutations(range(3))
    }
    expected = (
        min(scored.values()) if convention == "MIN_PLUS" else max(scored.values())
    )
    assert optimum.value == CanonicalRational.from_integer_ratio(expected, 1)
    assert assignments == tuple(
        permutation for permutation, score in scored.items() if score == expected
    )


def test_matrix_power_rejects_native_non_integer_exponents() -> None:
    with pytest.raises(OperationDomainValidationError):
        tropical_matrix_power(_matrix(), "1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("convention", "values", "expected", "shift"),
    [
        ("MIN_PLUS", (3, 8, None), (0, 5, None), -3),
        ("MAX_PLUS", (-7, 2, None), (-9, 0, None), -2),
    ],
)
def test_vector_projectivization_matches_exact_coordinate_oracle(
    convention: str,
    values: tuple[int | None, ...],
    expected: tuple[int | None, ...],
    shift: int,
) -> None:
    semiring = TropicalSemiring(convention=convention, base="ZZ")  # type: ignore[arg-type]
    infinity = "POSITIVE_INFINITY" if convention == "MIN_PLUS" else "NEGATIVE_INFINITY"
    vector = TropicalVector(
        semiring=semiring,
        axis=("a", "b", "c"),
        entries=tuple(
            TropicalScalar(
                semiring=semiring,
                kind="FINITE" if value is not None else infinity,
                value=CanonicalRational.from_integer_ratio(value, 1)
                if value is not None
                else None,
            )
            for value in values
        ),
    )

    kind, representative, translation = tropical_vector_projectivize(vector)

    assert kind == "PROJECTIVIZED"
    assert representative is not None and translation is not None
    actual = tuple(
        entry.value.as_fraction() if entry.value is not None else None
        for entry in representative.entries
    )
    assert actual == expected
    assert translation.value is not None
    assert translation.value.as_fraction() == shift
    finite_values = [v for v in actual if v is not None]
    assert (min(finite_values) if convention == "MIN_PLUS" else max(finite_values)) == 0
    # Common translation preserves pairwise differences, independently of the
    # implementation's extremum selection.
    assert actual[1] - actual[0] == values[1] - values[0]  # type: ignore[operator]


def test_all_infinity_vector_has_no_projective_class() -> None:
    semiring = _semiring()
    infinity = TropicalScalar(semiring=semiring, kind="POSITIVE_INFINITY", value=None)
    vector = TropicalVector(
        semiring=semiring, axis=("x", "y"), entries=(infinity, infinity)
    )

    kind, representative, translation = tropical_vector_projectivize(vector)

    assert (kind, representative, translation) == ("NO_PROJECTIVE_CLASS", None, None)
