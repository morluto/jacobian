"""Domain tests for exact Diophantine approximation operations."""

from __future__ import annotations

import decimal
import json
import math
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.diophantine_approximation import (
    _surd_kernel as surd_kernel,
)
from jacobian.math.number_theory.diophantine_approximation import (
    continued_fraction,
    convergents,
    nearest_integer_distance,
    range_profile,
    record_minima,
    scaled_floor,
    simultaneous_product,
    solve_pell,
)
from jacobian.math.number_theory.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ConvergentRequest,
    PellEquationRequest,
)
from jacobian.math.number_theory.diophantine_approximation._surd_models import (
    NearestIntegerDistanceRequest,
    NearestIntegerDistanceValue,
    RangeProfileRequest,
    RangeProfileResult,
    RecordMinimaRequest,
    RecordMinimaResult,
    ScaledFloorRequest,
    ScaledFloorValue,
    SimultaneousProductRequest,
    SimultaneousProductResult,
)
from jacobian.math.number_theory.diophantine_approximation._tools import (
    compute_continued_fraction,
    compute_convergents,
    compute_pell_equation,
)


def test_continued_fraction_sqrt_2() -> None:
    """sqrt(2) = [1; 2, 2, 2, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    )
    assert result.coefficients == (1, 2, 2, 2, 2)
    assert result.preperiod_length == 1
    assert result.period_length == 1


def test_continued_fraction_sqrt_3() -> None:
    """sqrt(3) = [1; 1, 2, 1, 2, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=6)
    )
    assert result.coefficients == (1, 1, 2, 1, 2, 1)
    assert result.preperiod_length == 1
    assert result.period_length == 2


def test_continued_fraction_sqrt_5() -> None:
    """sqrt(5) = [2; 4, 4, 4, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=5, term_count=5)
    )
    assert result.coefficients[0] == 2
    assert all(c == 4 for c in result.coefficients[1:])


def test_continued_fraction_expands_period_to_max_terms() -> None:
    """A one-term period still produces every requested coefficient."""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=500)
    )
    assert len(result.coefficients) == 500
    assert result.coefficients[0] == 1
    assert all(c == 2 for c in result.coefficients[1:])


def test_convergents_sqrt_2() -> None:
    """Convergents of sqrt(2): 1/1, 3/2, 7/5, 17/12, 41/29."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=5))
    assert len(result.convergents) == 5
    nums = [c.numerator for c in result.convergents]
    dens = [c.denominator for c in result.convergents]
    assert nums == [1, 3, 7, 17, 41]
    assert dens == [1, 2, 5, 12, 29]


def test_convergents_repeat_period_beyond_fixed_window() -> None:
    """Regression: a period of length one must expand for any convergent count."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=12))
    assert [c.index for c in result.convergents] == list(range(12))
    assert [c.numerator for c in result.convergents] == [
        1,
        3,
        7,
        17,
        41,
        99,
        239,
        577,
        1393,
        3363,
        8119,
        19601,
    ]
    assert [c.denominator for c in result.convergents] == [
        1,
        2,
        5,
        12,
        29,
        70,
        169,
        408,
        985,
        2378,
        5741,
        13860,
    ]


def test_convergents_expand_to_max_count() -> None:
    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=500)
    )
    assert len(result.convergents) == 500
    assert [c.index for c in result.convergents] == list(range(500))


def test_convergents_are_best_approximations() -> None:
    """Each convergent p/q satisfies |p^2 - D*q^2| < 2*sqrt(D)."""
    discriminant = 2
    result = compute_convergents(
        ConvergentRequest(discriminant=discriminant, convergent_count=10)
    )
    for conv in result.convergents:
        p = conv.numerator
        q = conv.denominator
        assert abs(p**2 - discriminant * q**2) < 2 * math.sqrt(discriminant)


def test_pell_equation_sqrt_2() -> None:
    """x^2 - 2*y^2 = 1 has fundamental solution (3, 2)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=2))
    assert result.x == 3
    assert result.y == 2
    assert result.x**2 - 2 * result.y**2 == 1


def test_pell_equation_sqrt_3() -> None:
    """x^2 - 3*y^2 = 1 has fundamental solution (2, 1)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=3))
    assert result.x == 2
    assert result.y == 1
    assert result.x**2 - 3 * result.y**2 == 1


def test_pell_equation_sqrt_5() -> None:
    """x^2 - 5*y^2 = 1 has fundamental solution (9, 4)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=5))
    assert result.x == 9
    assert result.y == 4
    assert result.x**2 - 5 * result.y**2 == 1


def test_pell_equation_sqrt_13() -> None:
    """x^2 - 13*y^2 = 1 has fundamental solution (649, 180)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=13))
    assert result.x == 649
    assert result.y == 180
    assert result.x**2 - 13 * result.y**2 == 1


def test_pell_equation_all_verified() -> None:
    """Every Pell solution satisfies x^2 - D*y^2 = 1."""
    for discriminant in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        result = compute_pell_equation(PellEquationRequest(discriminant=discriminant))
        x = result.x
        y = result.y
        assert x**2 - discriminant * y**2 == 1


def test_pell_equation_large_discriminant() -> None:
    """The derived period bound reaches a large fundamental solution exactly."""
    result = compute_pell_equation(PellEquationRequest(discriminant=991))
    x = result.x
    y = result.y
    assert x**2 - 991 * y**2 == 1


def test_pell_equation_long_period() -> None:
    """The longest period below the bound still reaches the fundamental solution."""
    result = compute_pell_equation(PellEquationRequest(discriminant=9949))
    x = result.x
    y = result.y
    assert x**2 - 9949 * y**2 == 1


def test_contract_rejects_non_squarefree() -> None:
    with pytest.raises(ValueError, match="squarefree"):
        compute_continued_fraction(
            ContinuedFractionRequest(discriminant=8, term_count=5)
        )


def test_contract_rejects_perfect_square() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        compute_continued_fraction(
            ContinuedFractionRequest(discriminant=9, term_count=5)
        )


def test_contract_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ContinuedFractionRequest(discriminant=1, term_count=5)


def test_public_kernels_reject_perfect_square() -> None:
    with pytest.raises(OperationDomainValidationError, match="perfect square"):
        continued_fraction(4, 5)
    with pytest.raises(OperationDomainValidationError, match="perfect square"):
        convergents(9, 3)
    with pytest.raises(OperationDomainValidationError, match="perfect square"):
        solve_pell(16)


@pytest.mark.parametrize(
    ("operation", "argument"),
    [(continued_fraction, 5_001), (convergents, 5_001)],
)
def test_public_kernels_reject_unbounded_prefix_requests(
    operation: Callable[[int, int], object], argument: int
) -> None:
    """Native callers share the materialized-prefix bound of the public models."""
    with pytest.raises(OperationDomainValidationError, match="between 1 and 5000"):
        operation(2, argument)


def test_public_kernels_reject_unbounded_discriminants() -> None:
    with pytest.raises(OperationDomainValidationError, match="between 2 and 1000000"):
        continued_fraction(1_000_001, 1)
    with pytest.raises(OperationDomainValidationError, match="between 2 and 1000000"):
        solve_pell(1_000_001)


def test_public_kernels_return_typed_values() -> None:
    fraction = continued_fraction(2, 3)
    assert fraction.coefficients == (1, 2, 2)
    assert (fraction.preperiod_length, fraction.period_length) == (1, 1)

    values = convergents(2, 3)
    assert [
        (value.index, value.numerator, value.denominator)
        for value in values.convergents
    ] == [(0, 1, 1), (1, 3, 2), (2, 7, 5)]

    pell = solve_pell(2)
    assert (pell.x, pell.y) == (3, 2)


# ---------------------------------------------------------------------------
# Explicit source-bound verifier regressions (#2313)
# ---------------------------------------------------------------------------


def test_continued_fraction_result_replays_known_answers() -> None:
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ContinuedFractionResult,
    )

    sqrt2 = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    )
    assert sqrt2.coefficients == (1, 2, 2, 2, 2)
    assert (sqrt2.preperiod_length, sqrt2.period_length) == (1, 1)
    parsed = ContinuedFractionResult.model_validate(sqrt2.model_dump())
    assert parsed == sqrt2

    sqrt3 = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=6)
    )
    assert sqrt3.coefficients == (1, 1, 2, 1, 2, 1)
    assert (sqrt3.preperiod_length, sqrt3.period_length) == (1, 2)

    one_period = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=2)
    )
    assert one_period.coefficients == (1, 2)
    two_periods = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=5)
    )
    assert two_periods.coefficients == (1, 1, 2, 1, 2)


def test_continued_fraction_prefix_boundary_semantics() -> None:
    """A truncated window retains its requested count and replays exactly."""

    exact_window = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=3)
    )
    assert exact_window.term_count == 3
    assert exact_window.coefficients == (1, 1, 2)
    assert exact_window.preperiod_length + exact_window.period_length == 3

    beyond = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=4)
    )
    assert beyond.coefficients[-1] == 1


def test_continued_fraction_result_rejects_mutations() -> None:
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ContinuedFractionResult,
    )

    with pytest.raises(ValidationError):
        ContinuedFractionResult.model_validate(
            {
                "discriminant": 2,
                "coefficients": (99,),
                "preperiod_length": 7,
                "period_length": 8,
                "term_count": 15,
            }
        )
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    ).model_dump()

    forged_coefficients = dict(result)
    forged_coefficients["coefficients"] = (1, 2, 2, 2, 3)
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionResult.model_validate(forged_coefficients)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.coefficient_bound_exceeded"
    )

    count_mismatch = dict(result, term_count=4)
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionResult.model_validate(count_mismatch)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.coefficient_count_mismatch"
    )


def test_convergent_result_replays_recurrence_and_determinant() -> None:
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ConvergentResult,
    )

    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=6))
    assert [(c.index, c.numerator, c.denominator) for c in result.convergents] == [
        (0, 1, 1),
        (1, 3, 2),
        (2, 7, 5),
        (3, 17, 12),
        (4, 41, 29),
        (5, 99, 70),
    ]
    parsed = [(c.numerator, c.denominator) for c in result.convergents]
    for n in range(1, len(parsed)):
        p_n, q_n = parsed[n]
        p_prev, q_prev = parsed[n - 1]
        determinant = p_n * q_prev - p_prev * q_n
        assert determinant == (-1) ** (n - 1)
    parsed_result = ConvergentResult.model_validate(result.model_dump())
    assert parsed_result == result

    sqrt3 = compute_convergents(ConvergentRequest(discriminant=3, convergent_count=4))
    parsed3 = [(c.numerator, c.denominator) for c in sqrt3.convergents]
    assert parsed3[:2] == [(1, 1), (2, 1)]


def test_convergent_result_rejects_mutations() -> None:
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ConvergentResult,
        ConvergentValue,
    )

    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult(
            discriminant=2,
            convergent_count=1,
            convergents=(ConvergentValue(index=77, numerator=0, denominator=0),),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.indices_not_contiguous"
    )
    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=4)
    ).model_dump()

    count_mismatch = dict(result, convergent_count=9)
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate(count_mismatch)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.convergent_count_mismatch"
    )


def test_convergent_result_rejects_oversized_components_before_bigint_work() -> None:
    """Forged long canonical strings die on the digit bound, before parsing/gcd.

    With convergent_count=4 the derived cap is
    ``_convergent_component_digit_cap(4)``, so a 100,000-digit numerator is
    rejected by string length alone; matching the digit-bound message proves
    the gate ran instead of the gcd/replay work.
    """
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ConvergentResult,
    )

    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=4)
    ).model_dump()

    long_numerator = dict(result)
    long_numerator["convergents"] = [dict(item) for item in result["convergents"]]
    long_numerator["convergents"][3]["numerator"] = "9" * 100_000
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate_json(json.dumps(long_numerator))
    assert exc_info.value.errors()[0]["type"] == "string_type"

    long_denominator = dict(result)
    long_denominator["convergents"] = [dict(item) for item in result["convergents"]]
    long_denominator["convergents"][3]["denominator"] = "7" * 100_000
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate_json(json.dumps(long_denominator))
    assert exc_info.value.errors()[0]["type"] == "string_type"


def test_convergent_digit_bound_admits_full_envelope() -> None:
    """Legitimate output across the admitted envelope stays inside the bound."""
    from jacobian.math.number_theory.diophantine_approximation._models import (
        ConvergentResult,
    )

    result = compute_convergents(
        ConvergentRequest(discriminant=9949, convergent_count=500)
    )
    assert ConvergentResult.model_validate(result.model_dump()) == result
    widest = max(
        max(len(str(abs(c.numerator))), len(str(abs(c.denominator))))
        for c in result.convergents
    )
    assert widest > 100


def test_producer_to_convergent_composition() -> None:
    """The CF coefficient stream composes into the serialized convergents."""

    cf = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=13, term_count=10)
    )
    convs = compute_convergents(ConvergentRequest(discriminant=13, convergent_count=10))
    coefficients = [cf.preperiod_length and x for x in cf.coefficients]
    p_prev2, p_prev1 = 1, coefficients[0]
    q_prev2, q_prev1 = 0, 1
    replayed = [(p_prev1, q_prev1)]
    for coefficient in coefficients[1:]:
        p_prev2, p_prev1 = p_prev1, coefficient * p_prev1 + p_prev2
        q_prev2, q_prev1 = q_prev1, coefficient * q_prev1 + q_prev2
        replayed.append((p_prev1, q_prev1))
    claimed = [(c.numerator, c.denominator) for c in convs.convergents]
    assert claimed == replayed


# ---------------------------------------------------------------------------
# Certified finite quadratic-surd and simultaneous approximation (#1786)
# ---------------------------------------------------------------------------


def _high_precision_sqrt(radicand: int, digits: int = 80) -> str:
    """Independent square root, computed without the module under test."""

    return format(Decimal(radicand).sqrt(context=decimal.Context(prec=digits)), "f")


def test_scaled_floor_matches_integer_square_definition() -> None:
    """floor(n*sqrt(d)) is isqrt(d*n^2) for a nonsquare radicand."""
    for multiplier, radicand in ((3, 2), (1, 2), (7, 3), (10, 5), (12, 6)):
        value = scaled_floor(
            ScaledFloorRequest(multiplier=multiplier, radicand=radicand)
        )
        assert value.floor == math.isqrt(radicand * multiplier * multiplier)
        assert value.ceiling == value.floor + 1
        assert value.square_lower == value.floor * value.floor
        assert value.square_upper == value.ceiling * value.ceiling


def test_scaled_floor_rejects_square_radicand() -> None:
    """A perfect-square radicand is outside the irrational contract."""
    with pytest.raises(OperationDomainValidationError):
        scaled_floor(ScaledFloorRequest(multiplier=2, radicand=9))


def test_nearest_integer_distance_branches_on_both_sides() -> None:
    """The nearest integer can come from either endpoint of the bracket."""
    floor_side = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=1, radicand=2, scale_bits=48)
    )
    ceiling_side = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=1, radicand=3, scale_bits=48)
    )
    assert floor_side.side == "FLOOR"
    assert floor_side.nearest_integer == floor_side.floor
    assert ceiling_side.side == "CEILING"
    assert ceiling_side.nearest_integer == ceiling_side.ceiling


def test_nearest_integer_distance_encloses_the_true_distance() -> None:
    """Independent high-precision square roots must lie inside the enclosure."""
    for multiplier, radicand in ((1, 2), (2, 3), (5, 7), (9, 11), (31, 13)):
        value = nearest_integer_distance(
            NearestIntegerDistanceRequest(
                multiplier=multiplier, radicand=radicand, scale_bits=64
            )
        )
        true_distance = abs(
            Decimal(multiplier) * Decimal(_high_precision_sqrt(radicand))
            - Decimal(value.nearest_integer)
        )
        assert value.distance_enclosure.lower.as_fraction() <= Fraction(true_distance)
        assert Fraction(true_distance) <= value.distance_enclosure.upper.as_fraction()


def test_insufficient_precision_is_reported_not_guessed() -> None:
    """Overlapping branches must be rejected, never tie-broken arbitrarily."""
    with pytest.raises(OperationDomainValidationError) as error:
        nearest_integer_distance(
            NearestIntegerDistanceRequest(multiplier=1, radicand=2, scale_bits=1)
        )
    assert error.value.errors()[0]["type"] == (
        "diophantine.nearest_integer_branch_unresolved"
    )


def test_refinement_tightens_without_reversing_a_separated_order() -> None:
    """A larger scale subdivides the same bracket and never reverses order."""
    coarse = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=1, radicand=2, scale_bits=8)
    )
    fine = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=1, radicand=2, scale_bits=64)
    )
    assert coarse.side == fine.side == "FLOOR"
    coarse_width = (
        coarse.distance_enclosure.upper.as_fraction()
        - coarse.distance_enclosure.lower.as_fraction()
    )
    fine_width = (
        fine.distance_enclosure.upper.as_fraction()
        - fine.distance_enclosure.lower.as_fraction()
    )
    assert fine_width <= coarse_width
    assert fine.distance_enclosure.lower.as_fraction() >= (
        coarse.distance_enclosure.lower.as_fraction()
    )
    assert fine.distance_enclosure.upper.as_fraction() <= (
        coarse.distance_enclosure.upper.as_fraction()
    )


def test_product_enclosure_includes_the_outer_factor_and_contains_the_value() -> None:
    """n * prod_i ||n sqrt(d_i)|| is enclosed, not the bare product of distances."""
    result = simultaneous_product(
        SimultaneousProductRequest(multiplier=4, radicands=(2, 3), scale_bits=64)
    )
    assert tuple(f.radicand for f in result.factors) == (2, 3)
    true_product = Decimal(4)
    for radicand in (2, 3):
        per_factor = nearest_integer_distance(
            NearestIntegerDistanceRequest(
                multiplier=4, radicand=radicand, scale_bits=64
            )
        )
        true_product *= abs(
            Decimal(4) * Decimal(_high_precision_sqrt(radicand))
            - Decimal(per_factor.nearest_integer)
        )
    assert result.product_enclosure.lower.as_fraction() <= Fraction(true_product)
    assert Fraction(true_product) <= result.product_enclosure.upper.as_fraction()


def test_product_enclosure_is_radicand_permutation_invariant() -> None:
    """Reordering the radicand axis permutes rows but preserves the product."""
    forward = simultaneous_product(
        SimultaneousProductRequest(multiplier=3, radicands=(2, 3), scale_bits=48)
    )
    reversed_axis = simultaneous_product(
        SimultaneousProductRequest(multiplier=3, radicands=(3, 2), scale_bits=48)
    )
    assert forward.product_enclosure == reversed_axis.product_enclosure
    assert tuple(f.radicand for f in reversed_axis.factors) == (3, 2)


def test_range_profile_accounts_for_every_integer() -> None:
    """The completed range covers each multiplier exactly once, in order."""
    result = range_profile(
        RangeProfileRequest(radicands=(2, 3), limit=25, scale_bits=40)
    )
    assert tuple(row.multiplier for row in result.rows) == tuple(range(1, 26))
    assert len(result.rows) == 25
    for row in result.rows:
        assert tuple(f.radicand for f in row.factors) == (2, 3)


def test_range_profile_rejects_rather_than_omitting_an_unresolved_row() -> None:
    """Completeness is all-or-nothing: a short profile is never returned."""
    with pytest.raises(OperationDomainValidationError) as error:
        range_profile(RangeProfileRequest(radicands=(2, 3), limit=5, scale_bits=2))
    assert error.value.errors()[0]["type"] == (
        "diophantine.range_profile_unresolved_row"
    )


def test_range_profile_rejects_aggregate_output_before_expansion() -> None:
    """A maximal table is refused before allocating its rows or intervals."""
    with pytest.raises(OperationDomainValidationError) as error:
        range_profile(
            RangeProfileRequest(
                radicands=(2, 3, 5, 6, 7, 10, 11, 13),
                limit=4096,
                scale_bits=4096,
            )
        )
    assert error.value.errors()[0]["type"] in {
        "diophantine.range_profile_output_bound",
        "diophantine.range_profile_intermediate_bound",
        "diophantine.range_profile_work_bound",
    }


def test_range_profile_admits_retained_allocation_before_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(surd_kernel, "MAX_SURD_RANGE_ALLOCATION_UNITS", 1)
    with pytest.raises(OperationResourceAdmissionError, match="allocation bound"):
        range_profile(RangeProfileRequest(radicands=(2, 3), limit=2, scale_bits=32))


def test_serialized_range_rejects_forged_factor_axes() -> None:
    result = range_profile(
        RangeProfileRequest(radicands=(2, 3), limit=2, scale_bits=32)
    )
    payload = result.model_dump(mode="json")
    payload["rows"][0]["factors"][0]["multiplier"] = "99"
    with pytest.raises(ValidationError):
        RangeProfileResult.model_validate_json(encode_strict_json(payload), strict=True)
    payload = result.model_dump(mode="json")
    payload["rows"][0]["factors"][0]["scale_bits"] = 31
    with pytest.raises(ValidationError):
        RangeProfileResult.model_validate_json(encode_strict_json(payload), strict=True)


def test_serialized_records_reject_forged_multiplier_axis() -> None:
    result = record_minima(
        RecordMinimaRequest(radicands=(2, 3), limit=20, scale_bits=64)
    )
    assert result.outcome == "COMPLETE"
    payload = result.model_dump(mode="json")
    payload["records"][0]["multiplier"] = "21"
    with pytest.raises(ValidationError):
        RecordMinimaResult.model_validate_json(encode_strict_json(payload), strict=True)


def test_record_minima_match_an_independent_decimal_record_oracle() -> None:
    """The record sequence agrees with a Decimal oracle, not this kernel."""
    result = record_minima(
        RecordMinimaRequest(radicands=(2, 3), limit=2000, scale_bits=64)
    )
    expected: list[int] = []
    incumbent: Decimal | None = None
    with decimal.localcontext() as context:
        context.prec = 180
        for multiplier in range(1, 2001):
            product = Decimal(multiplier)
            for radicand in (2, 3):
                root = Decimal(radicand).sqrt()
                floor = math.isqrt(radicand * multiplier * multiplier)
                distance = min(
                    Decimal(multiplier) * root - Decimal(floor),
                    Decimal(floor + 1) - Decimal(multiplier) * root,
                )
                product *= distance
            if incumbent is None or product < incumbent:
                expected.append(multiplier)
                incumbent = product
    assert result.outcome == "COMPLETE"
    assert tuple(row.multiplier for row in result.records) == tuple(expected)
    assert result.finite_argmin == expected[-1]


def test_record_minima_report_unresolved_without_an_argmin_claim() -> None:
    """A precision too coarse to separate a comparison makes no record claim.

    The exact distances are rational, so every branch eventually resolves; a
    scale that cannot resolve one row of the declared range must therefore
    reject rather than return a partial record sequence or a guessed argmin.
    """
    with pytest.raises(OperationDomainValidationError) as error:
        record_minima(RecordMinimaRequest(radicands=(2, 3), limit=40, scale_bits=1))
    assert error.value.errors()[0]["type"] == (
        "diophantine.range_profile_unresolved_row"
    )


def test_record_minima_resolved_scale_makes_only_strict_claims() -> None:
    """Every reported record is strictly below the incumbent it replaced."""
    result = record_minima(
        RecordMinimaRequest(radicands=(2, 3), limit=200, scale_bits=16)
    )
    assert result.outcome == "COMPLETE"
    assert result.finite_argmin == result.records[-1].multiplier
    for previous, current in zip(result.records, result.records[1:], strict=False):
        assert (
            current.product_enclosure.upper.as_fraction()
            < previous.product_enclosure.lower.as_fraction()
        )


def test_record_minima_reports_an_overlapping_nonrecord_comparison() -> None:
    """An overlapping candidate cannot be silently classified as a non-record."""
    result = record_minima(RecordMinimaRequest(radicands=(2, 3), limit=5, scale_bits=6))

    assert result.outcome == "UNRESOLVED"
    assert result.unresolved_multiplier == 5
    assert result.records[-1].multiplier == 4
    assert result.unresolved_product_enclosure is not None
    assert result.unresolved_incumbent_enclosure is not None
    assert (
        result.unresolved_product_enclosure.lower.as_fraction()
        < result.unresolved_incumbent_enclosure.upper.as_fraction()
    )


def test_scaled_floor_value_round_trips_through_strict_json() -> None:
    """The declared result survives strict JSON serialization unchanged."""
    result = scaled_floor(ScaledFloorRequest(multiplier=6, radicand=3))
    restored = ScaledFloorValue.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_nearest_integer_distance_round_trips_through_strict_json() -> None:
    """The certified enclosure survives strict JSON serialization unchanged."""
    result = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=5, radicand=7, scale_bits=32)
    )
    restored = NearestIntegerDistanceValue.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_serialized_surd_values_reject_forged_structural_fields() -> None:
    """Canonical carriers retain their domain and branch bindings on decode."""
    floor = scaled_floor(ScaledFloorRequest(multiplier=6, radicand=3))
    floor_payload = floor.model_dump(mode="json")
    floor_payload["radicand"] = 4
    with pytest.raises(ValidationError):
        ScaledFloorValue.model_validate_json(
            encode_strict_json(floor_payload), strict=True
        )

    distance = nearest_integer_distance(
        NearestIntegerDistanceRequest(multiplier=5, radicand=7, scale_bits=32)
    )
    distance_payload = distance.model_dump(mode="json")
    distance_payload["nearest_integer"] = distance.ceiling
    with pytest.raises(ValidationError):
        NearestIntegerDistanceValue.model_validate_json(
            encode_strict_json(distance_payload), strict=True
        )

    product = simultaneous_product(
        SimultaneousProductRequest(multiplier=1, radicands=(2, 3), scale_bits=32)
    )
    product_payload = product.model_dump(mode="json")
    product_payload["product_enclosure"]["lower"] = {"num": "-1", "den": "1"}
    with pytest.raises(ValidationError):
        SimultaneousProductResult.model_validate_json(
            encode_strict_json(product_payload), strict=True
        )


def test_serialized_record_result_rejects_missing_or_unbound_history() -> None:
    """A record result cannot lose its first row or incumbent source binding."""
    result = record_minima(
        RecordMinimaRequest(radicands=(2, 3), limit=20, scale_bits=64)
    )
    assert result.outcome == "COMPLETE"

    missing_first = result.model_dump(mode="json")
    missing_first["records"] = []
    with pytest.raises(ValidationError):
        RecordMinimaResult.model_validate_json(
            encode_strict_json(missing_first), strict=True
        )

    unbound_incumbent = result.model_dump(mode="json")
    unbound_incumbent["records"][1]["incumbent_enclosure"] = unbound_incumbent[
        "records"
    ][1]["product_enclosure"]
    with pytest.raises(ValidationError):
        RecordMinimaResult.model_validate_json(
            encode_strict_json(unbound_incumbent), strict=True
        )


def test_large_multiplier_uses_exact_json_integer_encoding() -> None:
    request = ScaledFloorRequest(multiplier=2**53, radicand=2)
    result = scaled_floor(request)
    restored = ScaledFloorValue.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_maximum_product_precision_fits_result_envelope() -> None:
    result = simultaneous_product(
        SimultaneousProductRequest(
            multiplier=1, radicands=(2, 3, 5, 6), scale_bits=4096
        )
    )
    restored = type(result).model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert restored == result
