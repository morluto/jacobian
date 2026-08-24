from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.math.matrices._operation_models import SquareRationalMatrixRequest
from jacobian.math.matrices._operations import compute_characteristic_polynomial
from jacobian.math.matrices.canonical_forms._models import (
    MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    MAX_MATRIX_POLYNOMIAL_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS,
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
)
from jacobian.math.matrices.canonical_forms._operations import (
    compute_matrix_polynomial_evaluation,
)
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(numerator: int | str, denominator: int | str = 1) -> CanonicalRational:
    return CanonicalRational(num=str(numerator), den=str(denominator))


def _matrix(*rows: tuple[int, ...]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(tuple(_rational(entry) for entry in row) for row in rows)
    )


def _polynomial(*terms: tuple[int | str, int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=(exponent,),
                )
                for coefficient, exponent in terms
            )
        ),
    )


def _fractions(matrix: RationalMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(entry.as_fraction() for entry in row) for row in matrix.entries)


def test_rotation_matrix_is_annihilated_by_t_squared_plus_one() -> None:
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, -1), (1, 0)),
        polynomial=_polynomial((1, 2), (1, 0)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert result.source_matrix == request.matrix
    assert result.polynomial == request.polynomial
    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 16
    assert result.method == "HORNER_OVER_QQ"


@pytest.mark.parametrize(
    ("polynomial", "expected"),
    [
        (_polynomial(), ((0, 0), (0, 0))),
        (_polynomial((3, 0)), ((3, 0), (0, 3))),
        (_polynomial((1, 1)), ((1, 2), (0, 1))),
    ],
)
def test_zero_constant_and_identity_polynomials(
    polynomial: RationalPolynomial,
    expected: tuple[tuple[int, ...], ...],
) -> None:
    result = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1, 2), (0, 1)),
            polynomial=polynomial,
        )
    )

    assert _fractions(result.value) == tuple(
        tuple(Fraction(entry) for entry in row) for row in expected
    )
    assert result.polynomial_degree == (
        None
        if not polynomial.polynomial.terms
        else polynomial.polynomial.terms[0].exponents[0]
    )


def test_evaluation_preserves_polynomial_sum_and_product() -> None:
    source = _matrix((1, 1), (0, 1))
    f = _polynomial((2, 1), (3, 0))
    g = _polynomial((1, 2), (-1, 0))
    polynomial_sum = _polynomial((1, 2), (2, 1), (2, 0))
    polynomial_product = _polynomial((2, 3), (3, 2), (-2, 1), (-3, 0))

    f_value = _fractions(
        compute_matrix_polynomial_evaluation(
            MatrixPolynomialEvaluationRequest(matrix=source, polynomial=f)
        ).value
    )
    g_value = _fractions(
        compute_matrix_polynomial_evaluation(
            MatrixPolynomialEvaluationRequest(matrix=source, polynomial=g)
        ).value
    )
    sum_value = _fractions(
        compute_matrix_polynomial_evaluation(
            MatrixPolynomialEvaluationRequest(
                matrix=source,
                polynomial=polynomial_sum,
            )
        ).value
    )
    product_value = _fractions(
        compute_matrix_polynomial_evaluation(
            MatrixPolynomialEvaluationRequest(
                matrix=source,
                polynomial=polynomial_product,
            )
        ).value
    )

    assert sum_value == tuple(
        tuple(f_value[row][column] + g_value[row][column] for column in range(2))
        for row in range(2)
    )
    assert product_value == tuple(
        tuple(
            sum(
                (f_value[row][inner] * g_value[inner][column] for inner in range(2)),
                start=Fraction(0),
            )
            for column in range(2)
        )
        for row in range(2)
    )


def test_value_composes_unchanged_with_matrix_consumers() -> None:
    evaluated = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((0, 1), (0, 0)),
            polynomial=_polynomial((1, 1), (1, 0)),
        )
    )

    characteristic = compute_characteristic_polynomial(
        SquareRationalMatrixRequest(matrix=evaluated.value)
    )

    assert tuple(
        coefficient.as_fraction()
        for coefficient in characteristic.coefficients_descending
    ) == (Fraction(1), Fraction(-2), Fraction(1))


def test_adapter_preserves_canonical_coefficients_above_python_digit_limit() -> None:
    numerator = "1" * 5_000
    result = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=_polynomial((numerator, 0)),
        )
    )

    assert result.value.entries[0][0].num == numerator
    assert result.value.entries[0][0].den == "1"


@pytest.mark.parametrize(
    "mutation",
    ["value", "matrix", "polynomial", "degree", "matrix_work", "scalar_work"],
)
def test_result_rejects_independent_source_value_and_work_mutations(
    mutation: str,
) -> None:
    result = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((0, -1), (1, 0)),
            polynomial=_polynomial((1, 2), (1, 0)),
        )
    )
    wire = deepcopy(result.model_dump(mode="json"))
    if mutation == "value":
        wire["value"]["entries"][0][0] = {"num": "1", "den": "1"}
    elif mutation == "matrix":
        wire["source_matrix"]["entries"][0][0] = {"num": "1", "den": "1"}
    elif mutation == "polynomial":
        wire["polynomial"]["polynomial"]["terms"][1]["coefficient"] = {
            "num": "2",
            "den": "1",
        }
    elif mutation == "degree":
        wire["polynomial_degree"] = 1
    elif mutation == "matrix_work":
        wire["matrix_multiplications"] = 1
    else:
        wire["scalar_product_terms"] = 1

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationResult.model_validate(wire)


def test_request_rejects_non_square_and_multivariate_sources() -> None:
    with pytest.raises(ValidationError, match="square matrix"):
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1, 2)),
            polynomial=_polynomial((1, 1)),
        )

    multivariate = RationalPolynomial(
        variables=("s", "t"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=_rational(1),
                    exponents=(1, 0),
                ),
            )
        ),
    )
    with pytest.raises(ValidationError, match="exactly one polynomial variable"):
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=multivariate,
        )


def test_horner_work_boundary_is_derived_from_degree_and_matrix_order() -> None:
    zero = _rational(0)
    one = _rational(1)
    identity = RationalMatrix(
        entries=tuple(
            tuple(one if row == column else zero for column in range(32))
            for row in range(32)
        )
    )

    maximum_degree = (
        MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS
        // MATRIX_POLYNOMIAL_EVALUATION_PASSES
        // 32**3
    )
    accepted = MatrixPolynomialEvaluationRequest(
        matrix=identity,
        polynomial=_polynomial((1, maximum_degree)),
    )
    assert accepted.polynomial.polynomial.terms[0].exponents == (maximum_degree,)

    with pytest.raises(ValidationError, match="scalar-product work bound"):
        MatrixPolynomialEvaluationRequest(
            matrix=identity,
            polynomial=_polynomial((1, maximum_degree + 1)),
        )


def test_work_admission_couples_products_to_exact_component_growth() -> None:
    moderate = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((10**30,)),
        polynomial=_polynomial((1, 500)),
    )
    assert moderate.polynomial.polynomial.terms[0].exponents == (500,)

    with pytest.raises(ValidationError, match="exact-arithmetic work"):
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((10**30,)),
            polynomial=_polynomial((1, 1_000)),
        )


def test_result_sensitive_admission_accepts_maximum_sparse_exponent_at_one_by_one() -> (
    None
):
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((1,)),
        polynomial=_polynomial((1, 32_768)),
    )
    assert request.polynomial.polynomial.terms[0].exponents == (32_768,)


def test_zero_matrix_admission_does_not_combine_irrelevant_denominators() -> None:
    first_denominator = "1" + "0" * 20_000
    second_denominator = format_canonical_integer(3**42_000)

    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0,)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(2,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(1,),
                    ),
                )
            ),
        ),
    )

    assert request.matrix.entries[0][0].num == "0"


def test_request_rejects_predicted_scalar_and_aggregate_output_overflow() -> None:
    denominator = "1" + "0" * 100
    overflowing_exponent = MAX_CANONICAL_RATIONAL_DIGITS // 100 + 1
    with pytest.raises(ValidationError, match="digit result bound"):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(1, denominator),),)),
            polynomial=_polynomial((1, overflowing_exponent)),
        )

    overflowing_entry_digits = CanonicalLimits().max_output_bytes // 32**2 + 1_000
    huge_coefficient = "1" * overflowing_entry_digits
    dense = RationalMatrix(
        entries=tuple(tuple(_rational(1) for _ in range(32)) for _ in range(32))
    )
    with pytest.raises(ValidationError, match="canonical output limit"):
        MatrixPolynomialEvaluationRequest(
            matrix=dense,
            polynomial=_polynomial((huge_coefficient, 1)),
        )


def test_structurally_nilpotent_powers_are_admitted_and_evaluate_to_zero() -> None:
    height = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(height)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 16


def test_structural_zero_reduces_to_surviving_terms_exactly() -> None:
    height = format_canonical_integer(7**18_000)
    nilpotent = RationalMatrix(
        entries=(
            (_rational(0), _rational(height)),
            (_rational(0), _rational(0)),
        )
    )
    with_constant = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=nilpotent,
            polynomial=_polynomial((1, 2), (5, 0)),
        )
    )
    assert _fractions(with_constant.value) == (
        (Fraction(5), Fraction(0)),
        (Fraction(0), Fraction(5)),
    )

    fractional = RationalMatrix(
        entries=((_rational(0), _rational(1, 3)), (_rational(0), _rational(0)))
    )
    rational_constant = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=fractional,
            polynomial=_polynomial((1, 2), (5, 0)),
        )
    )
    assert _fractions(rational_constant.value) == (
        (Fraction(5), Fraction(0)),
        (Fraction(0), Fraction(5)),
    )

    surviving_power = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=nilpotent,
            polynomial=_polynomial((1, 2), (1, 1)),
        )
    )
    assert surviving_power.value.entries[0][1].num == height
    assert surviving_power.value.entries[1][0].num == "0"


def test_admission_still_charges_live_structural_growth() -> None:
    height = "1" + "0" * 20_000
    cyclic = RationalMatrix(
        entries=(
            (_rational(0), _rational(height)),
            (_rational(height), _rational(0)),
        )
    )
    with pytest.raises(ValidationError, match="coefficient growth"):
        MatrixPolynomialEvaluationRequest(matrix=cyclic, polynomial=_polynomial((1, 2)))

    chain = RationalMatrix(
        entries=(
            (_rational(0), _rational(height), _rational(0)),
            (_rational(0), _rational(0), _rational(height)),
            (_rational(0), _rational(0), _rational(0)),
        )
    )
    with pytest.raises(ValidationError, match="coefficient growth"):
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 2)))

    vanishing_chain_value = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 3)))
    )
    assert _fractions(vanishing_chain_value.value) == (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
    )


def test_request_schema_publishes_coupled_digit_work_bound() -> None:
    schema = MatrixPolynomialEvaluationRequest.model_json_schema()

    polynomial_description = schema["properties"]["polynomial"]["description"]
    assert "(2 * degree * order^3) scalar products" in polynomial_description
    assert "largest decimal-digit component" in polynomial_description
    assert f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}" in polynomial_description


def _rational_polynomial(
    *terms: tuple[CanonicalRational, int],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=(exponent,),
                )
                for coefficient, exponent in terms
            )
        ),
    )


def test_degree_two_admission_cross_cancels_coefficient_and_matrix_power_factors() -> (
    None
):
    base_two = format_canonical_integer(2**53_179)
    base_three = format_canonical_integer(3**33_558)
    coefficient = _rational(
        format_canonical_integer(2**106_358),
        format_canonical_integer(3**67_116),
    )

    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(base_three, base_two),),)),
        polynomial=_rational_polynomial((coefficient, 2)),
    )
    result = compute_matrix_polynomial_evaluation(request)

    assert request.polynomial.polynomial.terms[0].exponents == (2,)
    assert result.value.entries[0][0].num == "1"
    assert result.value.entries[0][0].den == "1"
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2


def test_degree_two_admission_still_rejects_uncancellable_power_growth() -> None:
    entry_numerator = format_canonical_integer(7**24_048)
    entry_denominator = format_canonical_integer(5**28_072)
    coefficient_numerator = format_canonical_integer(11**18_900)
    coefficient_denominator = format_canonical_integer(13**17_400)

    with pytest.raises(ValidationError, match="digit result bound"):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=((_rational(entry_numerator, entry_denominator),),)
            ),
            polynomial=_rational_polynomial(
                (
                    _rational(coefficient_numerator, coefficient_denominator),
                    2,
                )
            ),
        )


def test_admission_falls_back_to_dense_bound_beyond_materialization_ceiling() -> None:
    height = "1" + "0" * 20_000
    with pytest.raises(ValidationError, match="digit result bound"):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(height),),)),
            polynomial=_polynomial((1, 5)),
        )

    moderate = "1" + "0" * 15_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(moderate),),)),
        polynomial=_polynomial((1, 2)),
    )
    assert request.matrix.entries[0][0].num == moderate


def test_structurally_dead_powers_are_excluded_from_digit_work_estimate() -> None:
    height = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(height)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 100)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert request.polynomial.polynomial.terms[0].exponents == (100,)
    assert result.polynomial_degree == 100
    assert result.matrix_multiplications == 100
    assert result.scalar_product_terms == 800
    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )


def _linear_rational_polynomial(
    coefficient: CanonicalRational,
    constant: CanonicalRational | None = None,
) -> RationalPolynomial:
    terms = [
        RationalPolynomialTerm(coefficient=coefficient, exponents=(1,)),
    ]
    if constant is not None:
        terms.append(RationalPolynomialTerm(coefficient=constant, exponents=(0,)))
    return RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(terms=tuple(terms)),
    )


def test_linear_admission_cross_cancels_rational_product_factors() -> None:
    numerator = format_canonical_integer(2**66_037)
    denominator = format_canonical_integer(3**42_017)
    matrix = RationalMatrix(entries=((_rational(denominator, numerator),),))
    coefficient = _rational(numerator, denominator)

    request = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_linear_rational_polynomial(coefficient),
    )
    result = compute_matrix_polynomial_evaluation(request)

    assert request.polynomial.polynomial.terms[0].exponents == (1,)
    assert result.value.entries[0][0].num == "1"
    assert result.value.entries[0][0].den == "1"

    cancelled_with_constant = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_linear_rational_polynomial(coefficient, _rational(5, 7)),
    )
    constant_result = compute_matrix_polynomial_evaluation(cancelled_with_constant)

    assert constant_result.value.entries[0][0].as_fraction() == Fraction(12, 7)


def test_linear_admission_still_rejects_uncancellable_product_growth() -> None:
    coefficient_numerator = format_canonical_integer(2**66_037)
    coefficient_denominator = format_canonical_integer(3**42_017)
    entry_numerator = format_canonical_integer(7**24_048)
    entry_denominator = format_canonical_integer(5**28_072)

    with pytest.raises(ValidationError, match="rational result bound"):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=((_rational(entry_numerator, entry_denominator),),)
            ),
            polynomial=_linear_rational_polynomial(
                _rational(coefficient_numerator, coefficient_denominator)
            ),
        )


def test_request_schema_publishes_coupled_degree_order_work_bound() -> None:
    schema = MatrixPolynomialEvaluationRequest.model_json_schema()

    polynomial_description = schema["properties"]["polynomial"]["description"]
    assert "2 * degree * order^3" in polynomial_description
    assert "4,000,000" in polynomial_description
    matrix_description = schema["properties"]["matrix"]["description"]
    assert "order 32" in matrix_description
