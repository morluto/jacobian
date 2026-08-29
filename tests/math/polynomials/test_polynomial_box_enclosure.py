"""Exact complete-box enclosure for canonical rational polynomials."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product

import pytest
from tests.math.polynomials._support import polynomial_validation_error

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.polynomials.intervals import polynomial_box_enclosure
from jacobian.math.polynomials.intervals._models import (
    MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS,
    MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS,
    MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS,
    MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE,
    MAX_BOX_ENCLOSURE_RESULT_BYTES,
    MAX_BOX_ENCLOSURE_RESULT_DIGITS,
    MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS,
    MAX_BOX_ENCLOSURE_TERMS,
    MAX_BOX_ENCLOSURE_TOTAL_DEGREE,
    PolynomialBoxEnclosureRequest,
    PolynomialBoxEnclosureResult,
    _estimate_growth,
)
from jacobian.math.polynomials.intervals._tools import (
    TOOLS,
    compute_polynomial_box_enclosure,
)
from jacobian.math.polynomials.maps.operations import jacobian_matrix
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _interval(lower: int | Fraction, upper: int | Fraction) -> ClosedRationalInterval:
    return ClosedRationalInterval(lower=_q(lower), upper=_q(upper))


def _box(
    variables: tuple[str, ...],
    intervals: tuple[tuple[int | Fraction, int | Fraction], ...],
) -> RationalBox:
    return RationalBox(
        variables=variables,
        intervals=tuple(_interval(lower, upper) for lower, upper in intervals),
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: Mapping[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_q(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _enclose(
    polynomial: RationalPolynomial, box: RationalBox
) -> PolynomialBoxEnclosureResult:
    return compute_polynomial_box_enclosure(
        PolynomialBoxEnclosureRequest(polynomial=polynomial, box=box)
    )


def _evaluate_at(
    polynomial: RationalPolynomial,
    point: tuple[Fraction, ...],
) -> Fraction:
    total = Fraction(0)
    for term in polynomial.polynomial.terms:
        value = term.coefficient.as_fraction()
        for coordinate, exponent in zip(point, term.exponents, strict=True):
            value *= coordinate**exponent
        total += value
    return total


def test_native_operation_matches_catalog_projection() -> None:
    polynomial = _polynomial(("x",), {(2,): 1, (0,): -1})
    box = _box(("x",), ((0, 2),))

    assert (
        polynomial_box_enclosure(polynomial, box) == _enclose(polynomial, box).enclosure
    )


def test_affine_and_multilinear_corner_extrema_are_enclosed_exactly() -> None:
    affine = _enclose(
        _polynomial(
            ("x", "y"),
            {(1, 0): 2, (0, 1): -3, (0, 0): 1},
        ),
        _box(("x", "y"), ((1, 2), (-1, 1))),
    )
    assert affine.enclosure == _interval(0, 8)

    multilinear = _enclose(
        _polynomial(("x", "y"), {(1, 1): 1}),
        _box(("x", "y"), ((-1, 2), (3, 4))),
    )
    assert multilinear.enclosure == _interval(-4, 8)


def test_zero_polynomial_and_point_box_have_exact_degenerate_values() -> None:
    box = _box(("x", "y"), ((-3, 5), (Fraction(1, 7), Fraction(2, 7))))
    zero = _enclose(_polynomial(("x", "y"), {}), box)
    assert zero.enclosure == _interval(0, 0)
    assert zero.box == box

    point = _box(
        ("x", "y"),
        ((Fraction(2, 3), Fraction(2, 3)), (Fraction(-1, 5), Fraction(-1, 5))),
    )
    evaluated = _enclose(
        _polynomial(
            ("x", "y"),
            {(2, 0): Fraction(3, 2), (0, 1): -2, (0, 0): Fraction(1, 7)},
        ),
        point,
    )
    exact = Fraction(3, 2) * Fraction(2, 3) ** 2 - 2 * Fraction(-1, 5) + Fraction(1, 7)
    assert evaluated.enclosure == _interval(exact, exact)
    assert evaluated.box.intervals[0].lower == evaluated.box.intervals[0].upper


def test_dependency_can_loosen_the_enclosure_without_claiming_a_root() -> None:
    result = _enclose(
        _polynomial(("x",), {(2,): 1, (1,): -1}),
        _box(("x",), ((0, 1),)),
    )

    # The exact range is [-1/4, 0], while the natural interval extension
    # treats the two occurrences of x independently. Soundness, not tightness,
    # is the public postcondition.
    assert result.enclosure == _interval(-1, 1)
    for numerator in range(9):
        x = Fraction(numerator, 8)
        value = x * x - x
        assert result.enclosure.lower.as_fraction() <= value
        assert value <= result.enclosure.upper.as_fraction()


def test_enclosure_contains_an_independently_evaluated_rational_grid() -> None:
    polynomial = _polynomial(
        ("x", "y"),
        {
            (3, 2): Fraction(3, 5),
            (2, 0): Fraction(-7, 4),
            (1, 1): 2,
            (0, 3): Fraction(-5, 6),
            (0, 0): Fraction(11, 13),
        },
    )
    box = _box(
        ("x", "y"),
        ((Fraction(-2, 3), Fraction(5, 4)), (Fraction(-3, 2), Fraction(2, 5))),
    )
    result = _enclose(polynomial, box).enclosure
    lower = result.lower.as_fraction()
    upper = result.upper.as_fraction()
    coordinate_samples = tuple(
        (
            interval.lower.as_fraction(),
            (interval.lower.as_fraction() + interval.upper.as_fraction()) / 2,
            interval.upper.as_fraction(),
        )
        for interval in box.intervals
    )

    for point in product(*coordinate_samples):
        value = _evaluate_at(polynomial, point)
        assert lower <= value <= upper


def test_small_exact_margin_composes_into_complete_box_exclusion() -> None:
    result = _enclose(
        _polynomial(
            ("x", "y"),
            {(2, 0): 1, (0, 2): 1, (0, 0): -1},
        ),
        _box(("x", "y"), ((2, 3), (0, 1))),
    )

    assert result.enclosure == _interval(3, 9)
    assert result.enclosure.lower.as_fraction() > 0


def test_jacobian_entry_is_consumed_without_parent_or_axis_conversion() -> None:
    source = _polynomial(
        ("x", "y"),
        {(2, 0): 1, (1, 1): 1, (0, 0): -1},
    )
    jacobian = jacobian_matrix(
        RationalPolynomialMap(
            input_variables=("x", "y"),
            output_polynomials=(source,),
        )
    )

    result = _enclose(
        jacobian.entries[0],
        _box(("x", "y"), ((1, 2), (3, 4))),
    )

    assert result.polynomial == jacobian.entries[0]
    assert result.enclosure == _interval(5, 8)


def test_transported_variable_and_axis_permutation_preserves_enclosure() -> None:
    source = _enclose(
        _polynomial(
            ("x", "y"),
            {(2, 1): 2, (1, 0): -1, (0, 0): Fraction(1, 3)},
        ),
        _box(("x", "y"), ((-1, 2), (3, 5))),
    )
    transported = _enclose(
        _polynomial(
            ("y", "x"),
            {(1, 2): 2, (0, 1): -1, (0, 0): Fraction(1, 3)},
        ),
        _box(("y", "x"), ((3, 5), (-1, 2))),
    )

    assert transported.enclosure == source.enclosure


def test_box_must_use_the_polynomial_complete_ordered_axis() -> None:
    polynomial = _polynomial(("x", "y"), {(1, 0): 1})
    request = PolynomialBoxEnclosureRequest(
        polynomial=polynomial,
        box=_box(("y", "x"), ((0, 1), (0, 1))),
    )
    with pytest.raises(OperationDomainValidationError):
        compute_polynomial_box_enclosure(request)


def test_reversed_coordinate_interval_is_rejected_before_execution() -> None:
    with polynomial_validation_error():
        ClosedRationalInterval(lower=_q(2), upper=_q(1))


@pytest.mark.parametrize(
    "mutation",
    ("polynomial", "box", "source_digest", "enclosure"),
)
def test_source_bound_result_rejects_invalid_source_binding(mutation: str) -> None:
    result = _enclose(
        _polynomial(("x",), {(1,): 1}),
        _box(("x",), ((1, 2),)),
    )
    payload = result.model_dump(mode="json")
    if mutation == "polynomial":
        payload["polynomial"]["polynomial"]["terms"][0]["coefficient"] = {
            "num": "2",
            "den": "1",
        }
    elif mutation == "box":
        payload["box"]["intervals"][0]["upper"] = {"num": "3", "den": "1"}
    elif mutation == "source_digest":
        payload["source_digest"] = "sha256:" + "0" * 64
    elif mutation == "enclosure":
        payload["enclosure"]["upper"] = {"num": "3", "den": "1"}

    if mutation == "enclosure":
        PolynomialBoxEnclosureResult.model_validate(payload)
    else:
        with polynomial_validation_error():
            PolynomialBoxEnclosureResult.model_validate(payload)


def test_digest_rejects_a_different_polynomial_with_the_same_interval() -> None:
    result = _enclose(
        _polynomial(("x",), {(1,): 1}),
        _box(("x",), ((0, 1),)),
    )
    payload = result.model_dump(mode="json")
    payload["polynomial"]["polynomial"]["terms"] = [
        {"coefficient": {"num": "-1", "den": "1"}, "exponents": [1]},
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
    ]

    with polynomial_validation_error():
        PolynomialBoxEnclosureResult.model_validate(payload)


def test_digest_rejects_a_different_box_when_the_polynomial_is_constant() -> None:
    result = _enclose(
        _polynomial(("x",), {}),
        _box(("x",), ((0, 1),)),
    )
    payload = result.model_dump(mode="json")
    payload["box"]["intervals"][0]["upper"] = {"num": "2", "den": "1"}

    with polynomial_validation_error():
        PolynomialBoxEnclosureResult.model_validate(payload)


def test_produced_result_round_trips_after_strict_serialization() -> None:
    result = _enclose(
        _polynomial(("x",), {(2,): 1, (0,): -2}),
        _box(("x",), ((1, 2),)),
    )

    assert (
        PolynomialBoxEnclosureResult.model_validate_json(
            result.model_dump_json(),
            strict=True,
        )
        == result
    )


def test_variable_count_boundary_preserves_axis_identity() -> None:
    variables = tuple(f"x{index}" for index in range(MAX_POLYNOMIAL_VARIABLES))
    request = PolynomialBoxEnclosureRequest(
        polynomial=_polynomial(variables, {}),
        box=_box(variables, tuple((0, 0) for _ in variables)),
    )
    assert request.box.variables == variables

    too_many = (*variables, "overflow")
    with polynomial_validation_error():
        RationalBox(
            variables=too_many,
            intervals=tuple(_interval(0, 0) for _ in too_many),
        )


def _many_exponents(count: int) -> tuple[tuple[int, int], ...]:
    side = MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE + 1
    return tuple(
        (
            MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE - index // side,
            MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE - index % side,
        )
        for index in range(count)
    )


def test_term_count_boundary_is_checked_before_interval_evaluation() -> None:
    assert MAX_BOX_ENCLOSURE_TERMS == MAX_POLYNOMIAL_TERMS
    variables = ("x", "y")
    at_limit = _polynomial(
        variables,
        dict.fromkeys(_many_exponents(MAX_BOX_ENCLOSURE_TERMS), 1),
    )
    PolynomialBoxEnclosureRequest(
        polynomial=at_limit,
        box=_box(variables, ((0, 0), (0, 0))),
    )

    with polynomial_validation_error():
        _polynomial(
            variables,
            dict.fromkeys(_many_exponents(MAX_BOX_ENCLOSURE_TERMS + 1), 1),
        )


def test_maximum_term_axis_work_completes_on_compact_unit_intermediates() -> None:
    variables = tuple(f"x{index}" for index in range(MAX_POLYNOMIAL_VARIABLES))
    terms = {
        (*exponents, *(0 for _ in range(MAX_POLYNOMIAL_VARIABLES - 2))): 1
        for exponents in _many_exponents(MAX_BOX_ENCLOSURE_TERMS)
    }
    result = _enclose(
        _polynomial(variables, terms),
        _box(variables, tuple((1, 1) for _ in variables)),
    )

    assert len(result.polynomial.polynomial.terms) * len(variables) == (
        MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS
    )
    assert result.enclosure == _interval(
        MAX_BOX_ENCLOSURE_TERMS,
        MAX_BOX_ENCLOSURE_TERMS,
    )


def test_per_variable_and_total_degree_boundaries() -> None:
    per_variable_limit = MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE
    total_limit = MAX_BOX_ENCLOSURE_TOTAL_DEGREE
    assert per_variable_limit == MAX_POLYNOMIAL_EXPONENT
    variables = tuple(f"x{index}" for index in range(MAX_POLYNOMIAL_VARIABLES))
    box = _box(variables, tuple((-1, 1) for _ in variables))
    boundary_exponents = (per_variable_limit,) * len(variables)
    assert sum(boundary_exponents) == total_limit

    result = _enclose(
        _polynomial(variables, {boundary_exponents: 1}),
        box,
    )
    assert result.enclosure == _interval(0, 1)
    with polynomial_validation_error():
        _polynomial(
            ("x",),
            {(per_variable_limit + 1,): 1},
        )


def test_coefficient_and_endpoint_digit_boundaries() -> None:
    assert (
        MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS
        == MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS
        == MAX_CANONICAL_RATIONAL_DIGITS
    )
    coefficient_at_limit = CanonicalRational(
        num="9" * MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS,
        den="1",
    )
    polynomial = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=coefficient_at_limit,
                    exponents=(0,),
                ),
            )
        ),
    )
    PolynomialBoxEnclosureRequest(
        polynomial=polynomial,
        box=_box(("x",), ((0, 0),)),
    )
    with polynomial_validation_error():
        CanonicalRational(
            num="9" * (MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS + 1),
            den="1",
        )

    endpoint_at_limit = CanonicalRational(
        num="9" * MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS,
        den="1",
    )
    PolynomialBoxEnclosureRequest(
        polynomial=_polynomial(("x",), {}),
        box=RationalBox(
            variables=("x",),
            intervals=(ClosedRationalInterval(lower=_q(0), upper=endpoint_at_limit),),
        ),
    )
    with polynomial_validation_error():
        CanonicalRational(
            num="9" * (MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS + 1),
            den="1",
        )


def _maximum_canonical_interval_payload() -> dict[str, dict[str, str]]:
    upper_integer = "9" * MAX_CANONICAL_RATIONAL_DIGITS
    lower_integer = f"{'9' * (MAX_CANONICAL_RATIONAL_DIGITS - 1)}8"
    return {
        "lower": {"num": lower_integer, "den": upper_integer},
        "upper": {"num": upper_integer, "den": lower_integer},
    }


def test_interval_ordering_at_the_canonical_boundary_is_admitted() -> None:
    assert MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS == 2 * MAX_CANONICAL_RATIONAL_DIGITS
    polynomial_payload = _polynomial(("x",), {}).model_dump(mode="json")
    request = PolynomialBoxEnclosureRequest.model_validate(
        {
            "polynomial": polynomial_payload,
            "box": {
                "variables": ["x"],
                "intervals": [_maximum_canonical_interval_payload()],
            },
        }
    )
    assert not request.polynomial.polynomial.terms

    oversized_interval = _maximum_canonical_interval_payload()
    oversized_interval["upper"]["num"] = "1" + "0" * MAX_CANONICAL_RATIONAL_DIGITS
    with polynomial_validation_error():
        PolynomialBoxEnclosureRequest.model_validate(
            {
                "polynomial": polynomial_payload,
                "box": {
                    "variables": ["x"],
                    "intervals": [oversized_interval],
                },
            }
        )


def test_structural_result_parse_accepts_canonical_forged_enclosure() -> None:
    result = _enclose(
        _polynomial(("x",), {}),
        _box(("x",), ((0, 0),)),
    )
    payload = result.model_dump(mode="json")
    payload["enclosure"] = _maximum_canonical_interval_payload()

    parsed = PolynomialBoxEnclosureResult.model_validate(payload)
    assert parsed.model_dump(mode="json") == payload


def _digit_rational(
    numerator_digits: int, denominator_digits: int
) -> CanonicalRational:
    return CanonicalRational(
        num="9" * numerator_digits,
        den="1" + "0" * (denominator_digits - 1),
    )


def _growth_boundary_request(
    *,
    second_endpoint_numerator_digits: int,
    second_endpoint_denominator_digits: int,
    coefficient_numerator: str,
    coefficient_denominator_digits: int,
) -> PolynomialBoxEnclosureRequest:
    variables = ("x", "y")
    degree = 64
    endpoint_digits = 128
    first = _digit_rational(
        endpoint_digits,
        endpoint_digits,
    )
    second = _digit_rational(
        second_endpoint_numerator_digits,
        second_endpoint_denominator_digits,
    )
    coefficient = CanonicalRational(
        num=coefficient_numerator,
        den="1" + "0" * (coefficient_denominator_digits - 1),
    )
    polynomial = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=(degree, degree),
                ),
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=(degree, degree - 1),
                ),
            )
        ),
    )
    box = RationalBox(
        variables=variables,
        intervals=(
            ClosedRationalInterval(lower=first, upper=first),
            ClosedRationalInterval(lower=second, upper=second),
        ),
    )
    return PolynomialBoxEnclosureRequest(polynomial=polynomial, box=box)


def test_exact_result_digit_growth_boundary() -> None:
    at_limit = _growth_boundary_request(
        second_endpoint_numerator_digits=128,
        second_endpoint_denominator_digits=127,
        coefficient_numerator="1",
        coefficient_denominator_digits=63,
    )
    assert MAX_BOX_ENCLOSURE_RESULT_DIGITS == 32_768
    assert compute_polynomial_box_enclosure(at_limit).box == at_limit.box

    above_limit = _growth_boundary_request(
        second_endpoint_numerator_digits=128,
        second_endpoint_denominator_digits=127,
        coefficient_numerator="1",
        coefficient_denominator_digits=64,
    )
    with pytest.raises(OperationDomainValidationError):
        compute_polynomial_box_enclosure(above_limit)


def test_intermediate_digit_growth_boundary() -> None:
    legacy_limit = _growth_boundary_request(
        second_endpoint_numerator_digits=127,
        second_endpoint_denominator_digits=127,
        coefficient_numerator="9" * 62,
        coefficient_denominator_digits=64,
    )
    assert (
        _estimate_growth(legacy_limit.polynomial, legacy_limit.box).intermediate_digits
        == 49_152
    )
    assert compute_polynomial_box_enclosure(legacy_limit).box == legacy_limit.box

    above_legacy_limit = _growth_boundary_request(
        second_endpoint_numerator_digits=127,
        second_endpoint_denominator_digits=127,
        coefficient_numerator="9" * 63,
        coefficient_denominator_digits=64,
    )
    growth = _estimate_growth(
        above_legacy_limit.polynomial,
        above_legacy_limit.box,
    )
    assert growth.intermediate_digits == 49_153
    assert growth.intermediate_digits < MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS
    assert (
        compute_polynomial_box_enclosure(above_legacy_limit).box
        == above_legacy_limit.box
    )


def test_result_byte_estimate_covers_exact_retained_source_serialization() -> None:
    polynomial = _polynomial(
        ("x", "y"),
        {
            (4, 1): Fraction(17, 19),
            (2, 3): Fraction(-23, 29),
            (0, 0): Fraction(31, 37),
        },
    )
    box = _box(
        ("x", "y"),
        ((Fraction(-41, 43), Fraction(47, 53)), (Fraction(59, 71), Fraction(67, 61))),
    )
    request = PolynomialBoxEnclosureRequest(polynomial=polynomial, box=box)
    result = compute_polynomial_box_enclosure(request)
    estimate = _estimate_growth(polynomial, box)
    serialized = encode_strict_json(result.model_dump(mode="json"))

    assert result.polynomial == polynomial
    assert result.box == box
    assert len(serialized) <= estimate.estimated_result_bytes
    assert estimate.estimated_result_bytes <= MAX_BOX_ENCLOSURE_RESULT_BYTES


def test_numeric_admission_limits_are_discoverable() -> None:
    tool = TOOLS[0]
    schema_description = tool.request_type.model_json_schema()["description"]
    for expected in (
        f"{MAX_BOX_ENCLOSURE_TERMS:,} terms",
        f"degree {MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE} per variable",
        f"{MAX_BOX_ENCLOSURE_TOTAL_DEGREE} total",
        f"{MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS}-digit coefficient",
        f"{MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS}-digit input-endpoint",
        f"{MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS:,} term-axis pairs",
        f"{MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS:,}-digit intermediate",
        f"{MAX_BOX_ENCLOSURE_RESULT_DIGITS:,}-digit result",
        f"{MAX_BOX_ENCLOSURE_RESULT_BYTES:,}-byte canonical retained-source",
    ):
        assert expected in schema_description
        assert expected in tool.description


def test_catalog_example_executes_the_declared_enclosure() -> None:
    tool = TOOLS[0]
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)

    assert tool.operation_id == "polynomial.box.enclosure.compute"
    assert result.enclosure == _interval(2, 6)
