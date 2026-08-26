"""Domain tests for exact Diophantine approximation operations."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.diophantine_approximation import (
    continued_fraction,
    convergents,
    solve_pell,
)
from jacobian.math.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ConvergentRequest,
    PellEquationRequest,
)
from jacobian.math.diophantine_approximation._operations import (
    compute_continued_fraction,
    compute_convergents,
    compute_pell_equation,
    verify_continued_fraction_result,
    verify_convergent_result,
    verify_pell_equation_result,
)


def test_continued_fraction_sqrt_2() -> None:
    """sqrt(2) = [1; 2, 2, 2, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    )
    assert result.coefficients == (1, 2, 2, 2, 2)
    assert result.preperiod_length == 1
    assert result.period_length == 1
    assert result.method == "SYMPY_CONTINUED_FRACTION"


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
    assert nums == ["1", "3", "7", "17", "41"]
    assert dens == ["1", "2", "5", "12", "29"]


def test_convergents_repeat_period_beyond_fixed_window() -> None:
    """Regression: a period of length one must expand for any convergent count."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=12))
    assert [c.index for c in result.convergents] == list(range(12))
    assert [c.numerator for c in result.convergents] == [
        "1",
        "3",
        "7",
        "17",
        "41",
        "99",
        "239",
        "577",
        "1393",
        "3363",
        "8119",
        "19601",
    ]
    assert [c.denominator for c in result.convergents] == [
        "1",
        "2",
        "5",
        "12",
        "29",
        "70",
        "169",
        "408",
        "985",
        "2378",
        "5741",
        "13860",
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
        p = parse_canonical_integer(conv.numerator)
        q = parse_canonical_integer(conv.denominator)
        assert abs(p**2 - discriminant * q**2) < 2 * math.sqrt(discriminant)


def test_pell_equation_sqrt_2() -> None:
    """x^2 - 2*y^2 = 1 has fundamental solution (3, 2)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=2))
    assert result.x == "3"
    assert result.y == "2"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 2 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_3() -> None:
    """x^2 - 3*y^2 = 1 has fundamental solution (2, 1)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=3))
    assert result.x == "2"
    assert result.y == "1"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 3 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_5() -> None:
    """x^2 - 5*y^2 = 1 has fundamental solution (9, 4)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=5))
    assert result.x == "9"
    assert result.y == "4"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 5 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_13() -> None:
    """x^2 - 13*y^2 = 1 has fundamental solution (649, 180)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=13))
    assert result.x == "649"
    assert result.y == "180"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 13 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_all_verified() -> None:
    """Every Pell solution satisfies x^2 - D*y^2 = 1."""
    for discriminant in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        result = compute_pell_equation(PellEquationRequest(discriminant=discriminant))
        x = parse_canonical_integer(result.x)
        y = parse_canonical_integer(result.y)
        assert x**2 - discriminant * y**2 == 1


def test_pell_equation_large_discriminant() -> None:
    """The derived period bound reaches a large fundamental solution exactly."""
    result = compute_pell_equation(PellEquationRequest(discriminant=991))
    x = parse_canonical_integer(result.x)
    y = parse_canonical_integer(result.y)
    assert x**2 - 991 * y**2 == 1


def test_pell_equation_long_period() -> None:
    """The longest period below the bound still reaches the fundamental solution."""
    result = compute_pell_equation(PellEquationRequest(discriminant=9949))
    x = parse_canonical_integer(result.x)
    y = parse_canonical_integer(result.y)
    assert x**2 - 9949 * y**2 == 1


def test_contract_rejects_non_squarefree() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionRequest(discriminant=4, term_count=5)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )


def test_contract_rejects_perfect_square() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionRequest(discriminant=9, term_count=5)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )


def test_contract_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ContinuedFractionRequest(discriminant=1, term_count=5)


def test_public_kernels_reject_perfect_square() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        continued_fraction(4, 5)
    with pytest.raises(ValueError, match="perfect square"):
        convergents(9, 3)
    with pytest.raises(ValueError, match="perfect square"):
        solve_pell(16)


def test_public_kernels_return_typed_values() -> None:
    assert continued_fraction(2, 3) == ([1, 2, 2], 1, 1)
    assert convergents(2, 3) == [(0, 1, 1), (1, 3, 2), (2, 7, 5)]
    assert solve_pell(2) == (3, 2)


# ---------------------------------------------------------------------------
# Explicit source-bound verifier regressions (#2313)
# ---------------------------------------------------------------------------


def test_continued_fraction_result_replays_known_answers() -> None:
    from jacobian.math.diophantine_approximation._models import (
        ContinuedFractionResult,
    )

    sqrt2 = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    )
    assert sqrt2.coefficients == (1, 2, 2, 2, 2)
    assert (sqrt2.preperiod_length, sqrt2.period_length) == (1, 1)
    parsed = ContinuedFractionResult.model_validate(sqrt2.model_dump())
    assert parsed == sqrt2
    assert verify_continued_fraction_result(parsed)

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
    from jacobian.math.diophantine_approximation._models import (
        ContinuedFractionResult,
    )

    with pytest.raises(ValidationError):
        ContinuedFractionResult(
            discriminant=2,
            coefficients=(99,),
            preperiod_length=7,
            period_length=8,
        )
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    ).model_dump()

    wrong_source = dict(result, discriminant=3)
    assert not verify_continued_fraction_result(
        ContinuedFractionResult.model_validate(wrong_source)
    )

    forged_coefficients = dict(result)
    forged_coefficients["coefficients"] = (1, 2, 2, 2, 3)
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionResult.model_validate(forged_coefficients)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.coefficient_bound_exceeded"
    )

    forged_metadata = dict(result, preperiod_length=2)
    assert not verify_continued_fraction_result(
        ContinuedFractionResult.model_validate(forged_metadata)
    )

    count_mismatch = dict(result, term_count=4)
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionResult.model_validate(count_mismatch)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.coefficient_count_mismatch"
    )


def test_convergent_result_replays_recurrence_and_determinant() -> None:
    from jacobian.math.diophantine_approximation._models import ConvergentResult

    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=6))
    assert [(c.index, c.numerator, c.denominator) for c in result.convergents] == [
        (0, "1", "1"),
        (1, "3", "2"),
        (2, "7", "5"),
        (3, "17", "12"),
        (4, "41", "29"),
        (5, "99", "70"),
    ]
    parsed = [
        (parse_canonical_integer(c.numerator), parse_canonical_integer(c.denominator))
        for c in result.convergents
    ]
    for n in range(1, len(parsed)):
        p_n, q_n = parsed[n]
        p_prev, q_prev = parsed[n - 1]
        determinant = p_n * q_prev - p_prev * q_n
        assert determinant == (-1) ** (n - 1)
    parsed_result = ConvergentResult.model_validate(result.model_dump())
    assert parsed_result == result
    assert verify_convergent_result(parsed_result)

    sqrt3 = compute_convergents(ConvergentRequest(discriminant=3, convergent_count=4))
    parsed3 = [
        (parse_canonical_integer(c.numerator), parse_canonical_integer(c.denominator))
        for c in sqrt3.convergents
    ]
    assert parsed3[:2] == [(1, 1), (2, 1)]


def test_convergent_result_rejects_mutations() -> None:
    from jacobian.math.diophantine_approximation._models import (
        ConvergentResult,
        ConvergentValue,
    )

    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult(
            discriminant=2,
            convergent_count=1,
            convergents=(ConvergentValue(index=77, numerator="0", denominator="0"),),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.indices_not_contiguous"
    )
    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=4)
    ).model_dump()

    zero_denominator = dict(result)
    zero_denominator["convergents"] = [dict(item) for item in result["convergents"]]
    zero_denominator["convergents"][0]["denominator"] = "0"
    assert not verify_convergent_result(
        ConvergentResult.model_validate(zero_denominator)
    )

    nonreduced = dict(result)
    nonreduced["convergents"] = [dict(item) for item in result["convergents"]]
    nonreduced["convergents"][1]["numerator"] = "6"
    nonreduced["convergents"][1]["denominator"] = "4"
    assert not verify_convergent_result(ConvergentResult.model_validate(nonreduced))

    recurrence_break = dict(result)
    recurrence_break["convergents"] = [dict(item) for item in result["convergents"]]
    recurrence_break["convergents"][2]["numerator"] = "8"
    assert not verify_convergent_result(
        ConvergentResult.model_validate(recurrence_break)
    )

    wrong_source = dict(result, discriminant=3)
    assert not verify_convergent_result(ConvergentResult.model_validate(wrong_source))

    count_mismatch = dict(result, convergent_count=9)
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate(count_mismatch)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.convergent_count_mismatch"
    )


def test_results_reject_non_squarefree_discriminant() -> None:
    """Results satisfy the same squarefree source domain as their requests.

    The forged payloads below carry the genuine canonical expansion and
    convergents of sqrt(8), so only the squarefree predicate can reject them.
    """
    from jacobian.math.diophantine_approximation._models import (
        ContinuedFractionResult,
        ConvergentResult,
        PellEquationResult,
    )

    # sqrt(8) = [2; overline{1, 4}] with convergents 2/1, 3/1, 14/5, 17/6.
    sqrt8_cf = {
        "discriminant": 8,
        "term_count": 5,
        "coefficients": (2, 1, 4, 1, 4),
        "preperiod_length": 1,
        "period_length": 2,
    }
    with pytest.raises(ValidationError) as exc_info:
        ContinuedFractionResult.model_validate(sqrt8_cf)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )

    sqrt8_convergents = {
        "discriminant": 8,
        "convergent_count": 4,
        "convergents": [
            {"index": 0, "numerator": "2", "denominator": "1"},
            {"index": 1, "numerator": "3", "denominator": "1"},
            {"index": 2, "numerator": "14", "denominator": "5"},
            {"index": 3, "numerator": "17", "denominator": "6"},
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate(sqrt8_convergents)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )

    with pytest.raises(ValidationError) as exc_info:
        PellEquationResult(discriminant=8, x="3", y="1")
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )

    with pytest.raises(ValidationError) as exc_info:
        PellEquationResult(discriminant=9, x="3", y="1")
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.discriminant_not_squarefree"
    )


def test_pell_result_verifier_rejects_nonfundamental_solution() -> None:
    from jacobian.math.diophantine_approximation._models import PellEquationResult

    result = compute_pell_equation(PellEquationRequest(discriminant=2))
    assert verify_pell_equation_result(result)
    # 17^2 - 2 * 12^2 = 1, but it is not the fundamental solution.
    assert not verify_pell_equation_result(
        PellEquationResult(discriminant=2, x="17", y="12")
    )


def test_convergent_result_rejects_oversized_components_before_bigint_work() -> None:
    """Forged long canonical strings die on the digit bound, before parsing/gcd.

    With convergent_count=4 the derived cap is
    ``_convergent_component_digit_cap(4)``, so a 100,000-digit numerator is
    rejected by string length alone; matching the digit-bound message proves
    the gate ran instead of the gcd/replay work.
    """
    from jacobian.math.diophantine_approximation._models import (
        ConvergentResult,
    )

    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=4)
    ).model_dump()

    long_numerator = dict(result)
    long_numerator["convergents"] = [dict(item) for item in result["convergents"]]
    long_numerator["convergents"][3]["numerator"] = "9" * 100_000
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate(long_numerator)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.component_digit_bound_exceeded"
    )

    long_denominator = dict(result)
    long_denominator["convergents"] = [dict(item) for item in result["convergents"]]
    long_denominator["convergents"][3]["denominator"] = "7" * 100_000
    with pytest.raises(ValidationError) as exc_info:
        ConvergentResult.model_validate(long_denominator)
    assert (
        exc_info.value.errors()[0]["type"]
        == "diophantine_approximation.component_digit_bound_exceeded"
    )


def test_convergent_digit_bound_admits_full_envelope() -> None:
    """Legitimate output across the admitted envelope stays inside the bound."""
    from jacobian.math.diophantine_approximation._models import ConvergentResult

    result = compute_convergents(
        ConvergentRequest(discriminant=9949, convergent_count=500)
    )
    assert ConvergentResult.model_validate(result.model_dump()) == result
    widest = max(
        max(len(c.numerator.lstrip("-")), len(c.denominator.lstrip("-")))
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
    claimed = [
        (parse_canonical_integer(c.numerator), parse_canonical_integer(c.denominator))
        for c in convs.convergents
    ]
    assert claimed == replayed
