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
    _MAX_WORK_BOUND,
    MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    MAX_MATRIX_POLYNOMIAL_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS,
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    _work_exact_quotient,
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

RationalInput = int | str | CanonicalRational | tuple[int | str, int | str]


def _rational(
    numerator: RationalInput, denominator: int | str = 1
) -> CanonicalRational:
    if isinstance(numerator, CanonicalRational):
        assert denominator == 1
        return numerator
    if isinstance(numerator, tuple):
        assert denominator == 1
        numerator, denominator = numerator
    return CanonicalRational(num=str(numerator), den=str(denominator))


def _matrix(*rows: tuple[RationalInput, ...]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(tuple(_rational(entry) for entry in row) for row in rows)
    )


def _polynomial(*terms: tuple[RationalInput, int]) -> RationalPolynomial:
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


def test_saturated_work_estimates_do_not_reenter_exact_division() -> None:
    assert _work_exact_quotient(84, 7) == 12
    assert _work_exact_quotient(_MAX_WORK_BOUND + 1, 7) > _MAX_WORK_BOUND


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
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
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

    with pytest.raises(ValidationError):
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

    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(1, denominator),),)),
            polynomial=_polynomial((1, overflowing_exponent)),
        )

    overflowing_entry_digits = CanonicalLimits().max_output_bytes // 32**2 + 1_000
    huge_coefficient = "1" * overflowing_entry_digits
    dense = RationalMatrix(
        entries=tuple(tuple(_rational(1) for _ in range(32)) for _ in range(32))
    )
    with pytest.raises(ValidationError):
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


def test_height_maximum_is_not_cancelled_from_the_result_bound() -> None:
    # diag(1, 1/q) with f = t^2 clears to Q = h = q, but h is only the
    # maximum cleared height, not a factor of every M^k entry: the true
    # second diagonal entry is 1/q^2 with compounded denominator digits.
    # The squared output must be predicted and rejected during request
    # validation instead of passing admission and failing result conversion.
    denominator = "1" + "0" * 17_000
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(1), _rational(0)),
                    (_rational(0), _rational(1, denominator)),
                )
            ),
            polynomial=_polynomial((1, 2)),
        )


def test_dead_powers_do_not_demand_a_global_clearing_denominator() -> None:
    # A square-zero matrix whose only entries carry coprime 17,000-digit
    # denominators: t^2 is structurally zero, so no global LCM of entry
    # denominators may be required before the support analysis runs.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(7**20_118)
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (
                    _rational(0),
                    _rational(1, first_denominator),
                    _rational(1, second_denominator),
                ),
                (_rational(0), _rational(0), _rational(0)),
                (_rational(0), _rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert _fractions(result.value) == tuple(
        tuple(Fraction(0) for _ in range(3)) for _ in range(3)
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 54


def test_proven_cancellations_survive_with_compounded_denominators() -> None:
    # Swapped denominators put the full compounded height into one cleared
    # row: h = q^2. With a matching lifted coefficient the proven factor
    # cancels q^2 exactly, so both requests stay admitted while the bounds
    # still charge the compounded denominators honestly.
    base = format_canonical_integer(2**12_000)
    swap = RationalMatrix(
        entries=(
            (_rational(0), _rational(base)),
            (_rational(1, base), _rational(0)),
        )
    )

    identity_result = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(matrix=swap, polynomial=_polynomial((1, 2)))
    )
    assert _fractions(identity_result.value) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )

    scaled_result = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=swap,
            polynomial=_polynomial((format_canonical_integer(2**24_000), 2)),
        )
    )
    assert scaled_result.value.entries[0][0].num == format_canonical_integer(2**24_000)
    assert scaled_result.value.entries[1][1].num == format_canonical_integer(2**24_000)
    assert scaled_result.value.entries[0][1].num == "0"


def test_unprovable_height_growth_is_rejected_during_request_validation() -> None:
    # Same swapped shape with a larger base and no cancellable coefficient:
    # the honest compounded prediction n * h^2 exceeds the canonical component
    # cap even though the exact value would be the identity. Admission cannot
    # establish the tighter claim, so the request must be rejected here rather
    # than admitted and rescued by result conversion.
    base = format_canonical_integer(2**40_000)
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(base)),
                    (_rational(1, base), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 2)),
        )


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
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(matrix=cyclic, polynomial=_polynomial((1, 2)))

    chain = RationalMatrix(
        entries=(
            (_rational(0), _rational(height), _rational(0)),
            (_rational(0), _rational(0), _rational(height)),
            (_rational(0), _rational(0), _rational(0)),
        )
    )
    with pytest.raises(ValidationError):
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
    assert "predicted shifted Horner intermediate components" in polynomial_description
    assert f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}" in polynomial_description


def _rational_polynomial(
    *terms: tuple[RationalInput, int],
) -> RationalPolynomial:
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

    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
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


def test_constant_result_work_estimates_are_not_clipped_at_the_component_cap() -> None:
    # A 4x4 superdiagonal chain of 32,768-digit entries with f = t^4: every
    # nonconstant power is structurally dead, yet Horner still materializes
    # an A^3 entry compounding three input heights (~98,304 digits) on the
    # way to the exact zero value. Clipping the shifted-height work proxy at
    # one canonical component would predict 32,769 digits and admit about
    # 5.5e11 digit-work units while execution performs about 4.9e12, so the
    # unclipped work estimate must reject this request during validation
    # while the same shape at heights whose compounded shifts fit the
    # coupled budget stays admitted.
    height = "1" + "0" * 32_767
    chain = RationalMatrix(
        entries=tuple(
            tuple(
                _rational(height) if column == row + 1 else _rational(0)
                for column in range(4)
            )
            for row in range(4)
        )
    )

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 4)))

    moderate_height = "1" + "0" * 13_999
    admitted = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (
                        _rational(0),
                        _rational(moderate_height),
                        _rational(0),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(moderate_height),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(moderate_height),
                    ),
                    (_rational(0), _rational(0), _rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 4)),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(4)) for _ in range(4)
    )
    assert admitted.matrix_multiplications == 4


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


def test_horner_charges_shifted_dead_leading_terms_during_their_ride() -> None:
    # The live t term keeps the square-zero support in the general branch,
    # and Horner materializes the doubled-height entry C*H on its first
    # multiplication before later shifts clear the structurally dead C*t^50.
    # Dropping the dead leading term from work accounting would admit this
    # request on a 20,001-digit prediction while execution constructs a
    # 40,000-digit product, so honest charging must reject it here while the
    # same shape at smaller heights stays admitted and evaluates exactly.
    height = "1" + "0" * 20_000
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(height)),
                    (_rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial(("1" + "0" * 20_000, 50), (1, 1)),
        )

    moderate_height = "1" + "0" * 14_999
    admitted = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(moderate_height)),
                    (_rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial(("1" + "0" * 14_999, 50), (1, 1)),
        )
    )

    assert admitted.value.entries[0][1].num == moderate_height
    assert admitted.value.entries[0][0].num == "0"
    assert admitted.polynomial_degree == 50
    assert admitted.matrix_multiplications == 50


def test_clearing_denominator_growth_is_bounded_by_live_horner_shifts() -> None:
    # Square-zero [[0, 1/q], [0, 0]] with a 20,001-digit q and f = t^100 + t:
    # the dead leading power clears on the first multiplication, so every
    # Horner intermediate stays 0, I, or A and no denominator ever exceeds
    # q. Raising the clearing denominator to the raw ordinary degree would
    # predict q^100 (~two million work digits) and reject a request whose
    # documented proxy 1600 * 20001^2 stays below the coupled budget;
    # bounding denominator growth by the maximum live Horner shift admits
    # it, and f(A) equals A exactly.
    denominator = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(1, denominator)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 100), (1, 1)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert result.value.entries[0][0].num == "0"
    assert result.value.entries[1][0].num == "0"
    assert result.value.entries[1][1].num == "0"
    assert result.value.entries[0][1].num == "1"
    assert result.value.entries[0][1].den == denominator
    assert result.polynomial_degree == 100
    assert result.matrix_multiplications == 100
    assert result.scalar_product_terms == 800


def test_overlapping_dead_denominators_are_charged_at_shared_entries() -> None:
    # Acyclic chain A = [[0,1,1],[0,0,1],[0,0,0]] with f = t^5/a + t^4/b for
    # coprime 32,768-digit denominators: both powers are structurally dead,
    # yet Horner temporarily forms (A^2)[0,2]/a + A[0,2]/b at the shared
    # entry [0,2], whose reduced denominator compounds to about 65,535
    # digits. Predicting only the largest single denominator admits about
    # 2.9e11 digit-work units while execution performs about 1.16e12, so the
    # resolved per-entry charge must reject this request while the same
    # shape at denominators whose compounded width fits the coupled budget
    # stays admitted and evaluates to the exact zero matrix.
    first_denominator = "1" + "0" * 32_767
    second_denominator = format_canonical_integer(11**31_400)
    chain = _matrix((0, 1, 1), (0, 0, 1), (0, 0, 0))

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, first_denominator), 5),
                (_rational(1, second_denominator), 4),
            ),
        )

    moderate_first = "1" + "0" * 16_383
    moderate_second = format_canonical_integer(11**15_700)
    admitted = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, moderate_first), 5),
                (_rational(1, moderate_second), 4),
            ),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(3)) for _ in range(3)
    )
    assert admitted.polynomial_degree == 5


def test_disjoint_dead_denominators_never_compound_across_entries() -> None:
    # Square-zero support with a live t term and f = t^3/a + t^2/b + t for
    # coprime 17,000-digit a and b: during the ride the 1/a term occupies
    # the off-diagonal while 1/b occupies the diagonal, and the former dies
    # before the latter shifts off-diagonal, so no entry ever combines the
    # denominators and every intermediate stays within 17,000 digits. A
    # global dead-term lcm would exceed the canonical cap and reject this
    # safely bounded request; resolved coexistence must admit it with
    # f(A) = A exactly.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, 1), (0, 0)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(3,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(2,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1),
                        exponents=(1,),
                    ),
                )
            ),
        ),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 3
    assert result.matrix_multiplications == 3


def test_mixed_overlap_of_dead_denominators_is_still_charged() -> None:
    # The disjointness relief above must not drop genuine overlap charging:
    # the same chain shape with a surviving t term rides t^5/a and t^4/b
    # through the shared entry [0,2] exactly as in the constant case, so
    # coprime 31,000-digit denominators compound past the digit-work budget
    # and must be rejected, while the smaller honest twin stays admitted
    # with its exact value preserved.
    first_denominator = "1" + "0" * 31_000
    second_denominator = format_canonical_integer(11**29_800)
    chain = _matrix((0, 1, 1), (0, 0, 1), (0, 0, 0))

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, first_denominator), 5),
                (_rational(1, second_denominator), 4),
                (_rational(1), 1),
            ),
        )

    moderate_first = "1" + "0" * 8_000
    moderate_second = format_canonical_integer(11**7_700)
    admitted = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, moderate_first), 5),
                (_rational(1, moderate_second), 4),
                (_rational(1), 1),
            ),
        )
    )

    assert _fractions(admitted.value) == (
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(0)),
    )


def test_converging_matrix_paths_compound_shared_cell_denominators() -> None:
    # A 4x4 diamond whose edges carry reciprocals of four pairwise-coprime
    # 25,000-digit integers with f = t^3: every power is structurally dead
    # and the exact value is zero, yet Horner materializes
    # (A^2)[0,3] = 1/(q1*q2) + 1/(q3*q4) at the shared cell [0,3], whose
    # reduced denominator compounds to about 100,000 digits. Predicting
    # entry_height^2 (about 50,000 digits) passes the coupled digit-work
    # check while the true intermediate exceeds it, so the walk-denominator
    # resolution must reject this request; the same shape at smaller
    # denominators stays admitted and evaluates exactly to zero.
    first = "1" + "0" * 24_999
    second = format_canonical_integer(3**52_408)
    third = format_canonical_integer(7**29_586)
    fourth = format_canonical_integer(11**24_007)
    diamond = RationalMatrix(
        entries=(
            (_rational(0), _rational(1, first), _rational(1, second), _rational(0)),
            (_rational(0), _rational(0), _rational(0), _rational(1, third)),
            (_rational(0), _rational(0), _rational(0), _rational(1, fourth)),
            (_rational(0), _rational(0), _rational(0), _rational(0)),
        )
    )

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=diamond,
            polynomial=_polynomial((1, 3)),
        )

    small_first = format_canonical_integer(3**12_578)
    small_second = format_canonical_integer(7**7_117)
    small_third = format_canonical_integer(11**5_834)
    small_fourth = format_canonical_integer(13**5_332)
    admitted = compute_matrix_polynomial_evaluation(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (
                        _rational(0),
                        _rational(1, small_first),
                        _rational(1, small_second),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(1, small_third),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(1, small_fourth),
                    ),
                    (_rational(0), _rational(0), _rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 3)),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(4)) for _ in range(4)
    )
    assert admitted.polynomial_degree == 3


def test_disjoint_rational_entries_never_demand_a_global_clearing_denominator() -> None:
    # Square-zero support with two rational entries in one row and a live t
    # term: f = t^2 + t with coprime 17,000-digit entry denominators keeps
    # the two denominators in separate entries at every Horner state and in
    # the exact value itself, so no per-cell coexistence ever compounds them.
    # Forming the global clearing lcm(a, b) exceeds the canonical cap and
    # would reject this request although every input, intermediate, output,
    # and digit-work bound holds; resolved coexistence admits it with
    # f(A) = A exactly.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (
                    _rational(0),
                    _rational(1, first_denominator),
                    _rational(1, second_denominator),
                ),
                (_rational(0), _rational(0), _rational(0)),
                (_rational(0), _rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2), (1, 1)),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert result.value.entries[0][1].num == "1"
    assert result.value.entries[0][1].den == first_denominator
    assert result.value.entries[0][2].den == second_denominator
    assert result.value.entries[1][0].num == "0"
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2


def test_dead_coefficient_powers_are_classified_before_the_coefficient_lcm() -> None:
    # Square-zero [[0, 1], [0, 0]] with f = t^3/a + t^2/b for coprime
    # 17,000-digit a and b: both nonconstant powers are structurally dead
    # and the exact value is zero, so forming lcm(a, b) before support
    # classification would reject a request whose every Horner intermediate
    # stays within a single input denominator.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, 1), (0, 0)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(3,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(2,),
                    ),
                )
            ),
        ),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 3
    assert result.matrix_multiplications == 3
    assert result.scalar_product_terms == 24


def test_surviving_coefficient_denominators_still_demand_a_common_denominator() -> None:
    # The same coprime denominators attached to powers that reach the value
    # compound genuinely: the linear identity case produces (a + b)/(ab) on
    # the diagonal, and the general degree-2 case clears its surviving
    # coefficients through lcm(a, b). Both compounded predictions exceed the
    # canonical cap, so admission must still reject during validation.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=RationalPolynomial(
                variables=("t",),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=_rational(1, first_denominator),
                            exponents=(1,),
                        ),
                        RationalPolynomialTerm(
                            coefficient=_rational(1, second_denominator),
                            exponents=(0,),
                        ),
                    )
                ),
            ),
        )

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((2, 0), (0, 2)),
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


def test_linear_admission_cross_cancels_rational_product_factors() -> None:
    numerator = format_canonical_integer(2**66_037)
    denominator = format_canonical_integer(3**42_017)
    matrix = RationalMatrix(entries=((_rational(denominator, numerator),),))
    coefficient = _rational(numerator, denominator)

    request = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_rational_polynomial((coefficient, 1)),
    )
    result = compute_matrix_polynomial_evaluation(request)

    assert request.polynomial.polynomial.terms[0].exponents == (1,)
    assert result.value.entries[0][0].num == "1"
    assert result.value.entries[0][0].den == "1"

    cancelled_with_constant = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_rational_polynomial((coefficient, 1), (_rational(5, 7), 0)),
    )
    constant_result = compute_matrix_polynomial_evaluation(cancelled_with_constant)

    assert constant_result.value.entries[0][0].as_fraction() == Fraction(12, 7)


def test_linear_admission_still_rejects_uncancellable_product_growth() -> None:
    coefficient_numerator = format_canonical_integer(2**66_037)
    coefficient_denominator = format_canonical_integer(3**42_017)
    entry_numerator = format_canonical_integer(7**24_048)
    entry_denominator = format_canonical_integer(5**28_072)

    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=((_rational(entry_numerator, entry_denominator),),)
            ),
            polynomial=_rational_polynomial(
                (_rational(coefficient_numerator, coefficient_denominator), 1)
            ),
        )


def test_linear_output_bounds_preserve_additive_cancellation() -> None:
    # [H] with f = t - H: both diagonal summands have magnitude H yet their
    # exact sum is zero, so magnitude-only addition charges 2H and rejects a
    # request whose digit work and exact value are small. Sign-preserving
    # reduction must admit the exact zero instead.
    height = "9" + "0" * 32_767
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(height),),)),
        polynomial=_rational_polynomial(
            (_rational(1), 1), (_rational("-" + height), 0)
        ),
    )

    result = compute_matrix_polynomial_evaluation(request)

    assert _fractions(result.value) == ((Fraction(0),),)
    assert result.polynomial_degree == 1
    assert result.matrix_multiplications == 1
    assert result.scalar_product_terms == 1


def test_linear_output_bounds_still_reject_uncancellable_additive_growth() -> None:
    # The additive twin without cancellation: f = t + H genuinely evaluates
    # to 2H, whose 32,769 digits exceed the canonical component cap, so the
    # reduced exact bound still rejects during request validation.
    height = "9" + "0" * 32_767
    with pytest.raises(ValidationError):
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(height),),)),
            polynomial=_rational_polynomial((_rational(1), 1), (_rational(height), 0)),
        )


def test_request_schema_publishes_coupled_degree_order_work_bound() -> None:
    schema = MatrixPolynomialEvaluationRequest.model_json_schema()

    polynomial_description = schema["properties"]["polynomial"]["description"]
    assert "2 * degree * order^3" in polynomial_description
    assert f"{MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,}" in polynomial_description
    matrix_description = schema["properties"]["matrix"]["description"]
    assert "order 32" in matrix_description
