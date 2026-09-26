from __future__ import annotations

from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import (
    MatrixAssignmentRequest,
    MatrixFinitePowerSumRequest,
    MatrixMinorAssignmentsRequest,
    MatrixMinorAssignmentsResult,
)
from jacobian.math.polynomials.tropical._tools import (
    TOOLS,
    compute_finite_power_sum,
)
from jacobian.math.polynomials.tropical.operations import (
    tropical_assignment_profile,
    tropical_matrix_minor_assignment_profiles,
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


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_minor_assignment_profiles_match_independent_subset_oracle(
    convention: str,
) -> None:
    semiring = TropicalSemiring(convention=convention, base="ZZ")  # type: ignore[arg-type]
    values = ((0, 0), (0, 0), (6, 4))
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=("r0", "r1", "r2"),
        column_axis=("c0", "c1"),
        entries=tuple(
            tuple(_scalar_for(semiring, value) for value in row) for row in values
        ),
    )

    request = MatrixMinorAssignmentsRequest(matrix=matrix, sizes=(1, 2))
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "tropical.matrix.minor_assignment_profiles.compute"
    )
    result = operation.run(request)
    assert result.matrix == matrix
    assert result.sizes == (1, 2)
    assert len(result.minors) == 9
    for profile in result.minors:
        size = len(profile.row_indices)
        scores = {
            permutation: sum(
                values[profile.row_indices[row]][profile.column_indices[column]]
                for row, column in enumerate(permutation)
            )
            for permutation in permutations(range(size))
        }
        expected = (
            min(scores.values()) if convention == "MIN_PLUS" else max(scores.values())
        )
        assert profile.value.value == CanonicalRational.from_integer_ratio(expected, 1)
        assert profile.permutations == tuple(
            permutation for permutation, score in scores.items() if score == expected
        )

    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result


def _scalar_for(semiring: TropicalSemiring, value: int) -> TropicalScalar:
    return TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(value, 1),
    )


def test_minor_assignment_profiles_accept_full_envelope_one_by_one() -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="QQ")
    value = 10**MAX_TROPICAL_SCALAR_DIGITS - 1
    scalar = TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(value, 1),
    )
    matrix = TropicalMatrix(
        semiring=semiring, row_axis=("r",), column_axis=("c",), entries=((scalar,),)
    )
    (profile,) = tropical_matrix_minor_assignment_profiles(matrix, (1,))
    assert profile.value == scalar


def test_minor_assignment_result_rejects_false_source_profile() -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=("r",),
        column_axis=("c",),
        entries=((_scalar_for(semiring, 2),),),
    )
    (profile,) = tropical_matrix_minor_assignment_profiles(matrix, (1,))
    payload = MatrixMinorAssignmentsResult(
        matrix=matrix, sizes=(1,), minors=(profile,)
    ).model_dump(mode="json")
    payload["minors"][0]["column_indices"] = [4]
    with pytest.raises(ValidationError):
        MatrixMinorAssignmentsResult.model_validate(payload)


def test_minor_assignment_native_boundary_classifies_unhashable_sizes() -> None:
    with pytest.raises(OperationDomainValidationError):
        tropical_matrix_minor_assignment_profiles(_matrix(), ([1],))  # type: ignore[arg-type]


def test_minor_assignment_profiles_preserve_all_infinite_ties() -> None:
    semiring = _semiring()
    infinity = TropicalScalar(semiring=semiring, kind="POSITIVE_INFINITY", value=None)
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=("r0", "r1"),
        column_axis=("c0", "c1"),
        entries=((infinity, infinity), (infinity, infinity)),
    )
    (profile,) = tropical_matrix_minor_assignment_profiles(matrix, (2,))
    assert profile.value.kind == "POSITIVE_INFINITY"
    assert profile.permutations == ((0, 1), (1, 0))


def test_minor_assignment_profiles_admit_one_order_eight_minor() -> None:
    semiring = _semiring()
    matrix = TropicalMatrix(
        semiring=semiring,
        row_axis=tuple(f"r{i}" for i in range(8)),
        column_axis=tuple(f"c{i}" for i in range(8)),
        entries=tuple(
            tuple(
                _scalar(CanonicalRational.from_integer_ratio(i + j, 1))
                for j in range(8)
            )
            for i in range(8)
        ),
    )
    (profile,) = tropical_matrix_minor_assignment_profiles(matrix, (8,))
    assert profile.value.value == CanonicalRational.from_integer_ratio(56, 1)
    assert len(profile.permutations) == 40_320


def test_minor_assignment_profiles_reject_noncanonical_native_axes() -> None:
    matrix = _matrix().model_copy(update={"row_axis": ("r", "r")})
    with pytest.raises(OperationDomainValidationError):
        tropical_matrix_minor_assignment_profiles(matrix, (1,))


def test_minor_assignment_profiles_admit_factorial_work_before_enumeration() -> None:
    matrix = TropicalMatrix(
        semiring=_semiring(),
        row_axis=tuple(f"r{i}" for i in range(8)),
        column_axis=tuple(f"c{i}" for i in range(8)),
        entries=tuple(
            tuple(_scalar(CanonicalRational.from_integer_ratio(0, 1)) for _ in range(8))
            for _ in range(8)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_matrix_minor_assignment_profiles(matrix, (4,))
    assert error.value.errors()[0]["type"] == "tropical.minor_assignment_work"


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
