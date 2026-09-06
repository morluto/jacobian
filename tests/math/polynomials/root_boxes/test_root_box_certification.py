"""Exact Krawczyk certification for square rational polynomial systems."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.matrices.operations import determinant_result
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.root_boxes import (
    certify_real_root_box,
    verify_real_root_box,
)
from jacobian.math.polynomials.root_boxes._models import (
    MAX_ROOT_BOX_DIMENSION,
    PolynomialSystemRootBoxRequest,
    PolynomialSystemRootBoxResult,
    RootBoxCertifiedUniqueNonsingular,
    RootBoxComponentExclusion,
    RootBoxInconclusiveKrawczykAttempt,
    RootBoxKrawczykDisjointness,
    RootBoxNoRoot,
    RootBoxSingularMidpointAttempt,
    RootBoxUnknown,
)
from jacobian.math.polynomials.root_boxes._tools import (
    TOOLS,
    compute_polynomial_system_root_box,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _interval(
    lower: int | Fraction,
    upper: int | Fraction,
) -> ClosedRationalInterval:
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


def _map(
    variables: tuple[str, ...],
    components: tuple[Mapping[tuple[int, ...], int | Fraction], ...],
) -> RationalPolynomialMap:
    return RationalPolynomialMap(
        input_variables=variables,
        output_polynomials=tuple(_polynomial(variables, terms) for terms in components),
    )


def _certify(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> PolynomialSystemRootBoxResult:
    return compute_polynomial_system_root_box(
        PolynomialSystemRootBoxRequest(polynomial_map=polynomial_map, box=box)
    )


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _matrix_product(
    left: tuple[tuple[CanonicalRational, ...], ...],
    right: tuple[tuple[CanonicalRational, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    _fraction(left[row][inner]) * _fraction(right[inner][column])
                    for inner in range(len(left))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def test_square_root_two_has_exact_replayable_krawczyk_certificate() -> None:
    polynomial_map = _map(("x",), ({(2,): 1, (0,): -2},))
    box = _box(("x",), ((1, 2),))

    result = _certify(polynomial_map, box)

    assert isinstance(result.conclusion, RootBoxCertifiedUniqueNonsingular)
    evidence = result.conclusion.evidence
    assert evidence.center.values == (_q(Fraction(3, 2)),)
    assert evidence.value_at_center == (_q(Fraction(1, 4)),)
    assert evidence.jacobian_at_center.entries == ((_q(3),),)
    assert evidence.preconditioner.entries == ((_q(Fraction(1, 3)),),)
    assert evidence.jacobian_enclosure.entries == (((_interval(2, 4)),),)
    assert evidence.krawczyk_image.intervals == (
        _interval(Fraction(5, 4), Fraction(19, 12)),
    )
    source = box.intervals[0]
    image = evidence.krawczyk_image.intervals[0]
    assert source.lower.as_fraction() < image.lower.as_fraction()
    assert image.upper.as_fraction() < source.upper.as_fraction()
    assert _matrix_product(
        evidence.preconditioner.entries,
        evidence.jacobian_at_center.entries,
    ) == ((Fraction(1),),)
    assert determinant_result(evidence.jacobian_at_center).determinant == _q(3)


def test_native_operation_matches_catalog_projection() -> None:
    polynomial_map = _map(("x",), ({(2,): 1, (0,): -2},))
    box = _box(("x",), ((1, 2),))

    assert certify_real_root_box(polynomial_map, box) == _certify(polynomial_map, box)


def test_component_interval_exclusion_proves_no_root_on_complete_box() -> None:
    result = _certify(
        _map(("x",), ({(2,): 1, (0,): 1},)),
        _box(("x",), ((-1, 1),)),
    )

    assert isinstance(result.conclusion, RootBoxNoRoot)
    assert isinstance(result.conclusion.evidence, RootBoxComponentExclusion)
    assert result.conclusion.evidence.component_index == 0
    assert result.conclusion.evidence.enclosure == _interval(1, 2)


def test_component_exclusion_skips_an_unneeded_expensive_system_range() -> None:
    exponent_pairs = tuple(
        (left, degree - left) for degree in range(9) for left in range(degree + 1)
    )
    small_primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
    )
    denominators: list[int] = []
    for prime in small_primes:
        denominator = 1
        while denominator * prime < 10**128:
            denominator *= prime
        denominators.append(denominator)
    expensive_component: dict[tuple[int, ...], int | Fraction] = {
        exponents: Fraction(1, denominator)
        for exponents, denominator in zip(
            exponent_pairs,
            denominators,
            strict=True,
        )
    }

    components: tuple[Mapping[tuple[int, ...], int | Fraction], ...] = (
        {(0, 0): 1},
        expensive_component,
    )
    result = _certify(
        _map(("x", "y"), components),
        _box(("x", "y"), ((0, 1), (0, 1))),
    )

    assert isinstance(result.conclusion, RootBoxNoRoot)
    assert isinstance(result.conclusion.evidence, RootBoxComponentExclusion)
    assert result.conclusion.evidence.component_index == 0
    assert result.conclusion.evidence.enclosure == _interval(1, 1)


def test_disjoint_krawczyk_image_proves_no_root_when_component_range_does_not() -> None:
    result = _certify(
        _map(("x",), ({(2,): 1, (1,): -3, (0,): 3},)),
        _box(("x",), ((2, 3),)),
    )

    assert isinstance(result.conclusion, RootBoxNoRoot)
    assert isinstance(result.conclusion.evidence, RootBoxKrawczykDisjointness)
    evidence = result.conclusion.evidence.evidence
    assert evidence.krawczyk_image.intervals == (
        _interval(Fraction(11, 8), Fraction(15, 8)),
    )
    assert evidence.krawczyk_image.intervals[0].upper.as_fraction() < 2


def test_singular_midpoint_is_an_explicit_non_conclusion() -> None:
    result = _certify(
        _map(("x",), ({(2,): 1},)),
        _box(("x",), ((-1, 1),)),
    )

    assert isinstance(result.conclusion, RootBoxUnknown)
    assert isinstance(result.conclusion.attempt, RootBoxSingularMidpointAttempt)
    data = result.conclusion.attempt.data
    assert data.value_at_center == (_q(0),)
    assert data.jacobian_at_center.entries == ((_q(0),),)
    assert data.jacobian_enclosure.entries == (((_interval(-2, 2)),),)


def test_tiny_residual_on_positive_dimensional_locus_is_not_a_root_claim() -> None:
    epsilon = Fraction(1, 10**20)
    result = _certify(
        _map(
            ("x", "y"),
            (
                {(1, 0): 1, (0, 1): -1},
                {(2, 0): 1, (1, 1): -2, (0, 2): 1},
            ),
        ),
        _box(("x", "y"), ((0, 2 * epsilon), (0, 4 * epsilon))),
    )

    assert isinstance(result.conclusion, RootBoxUnknown)
    assert isinstance(result.conclusion.attempt, RootBoxSingularMidpointAttempt)
    data = result.conclusion.attempt.data
    assert data.center.values == (_q(epsilon), _q(2 * epsilon))
    assert data.value_at_center == (_q(-epsilon), _q(epsilon * epsilon))
    assert determinant_result(data.jacobian_at_center).determinant == _q(0)


@pytest.mark.parametrize("endpoints", (((0, 1),), ((0, 0),)))
def test_boundary_and_point_roots_remain_unknown(
    endpoints: tuple[tuple[int, int], ...],
) -> None:
    result = _certify(
        _map(("x",), ({(1,): 1},)),
        _box(("x",), endpoints),
    )

    assert isinstance(result.conclusion, RootBoxUnknown)
    assert isinstance(result.conclusion.attempt, RootBoxInconclusiveKrawczykAttempt)
    assert result.conclusion.attempt.evidence.krawczyk_image.intervals == (
        _interval(0, 0),
    )


def test_five_variable_tiny_box_regression_certifies_a_coupled_system() -> None:
    variables = tuple(f"x{index}" for index in range(5))
    square_terms: dict[tuple[int, ...], int] = {}
    for left in range(5):
        for right in range(left, 5):
            exponents = tuple(
                int(axis == left) + int(axis == right) for axis in range(5)
            )
            square_terms[exponents] = 1 if left == right else 2
    components = []
    for output in range(5):
        terms = dict(square_terms)
        terms[tuple(int(axis == output) for axis in range(5))] = 1
        components.append(terms)
    radius = Fraction(1, 10**55)

    result = _certify(
        _map(variables, tuple(components)),
        _box(variables, tuple((-radius, radius) for _ in variables)),
    )

    assert isinstance(result.conclusion, RootBoxCertifiedUniqueNonsingular)
    evidence = result.conclusion.evidence
    expected_image = _interval(-50 * radius * radius, 50 * radius * radius)
    assert evidence.krawczyk_image.intervals == (expected_image,) * 5
    assert _matrix_product(
        evidence.preconditioner.entries,
        evidence.jacobian_at_center.entries,
    ) == tuple(
        tuple(Fraction(row == column) for column in range(5)) for row in range(5)
    )
    for interval in evidence.krawczyk_image.intervals:
        assert -radius < interval.lower.as_fraction()
        assert interval.upper.as_fraction() < radius


def test_five_variable_shifted_tiny_boxes_retain_shared_denominators() -> None:
    variables = tuple(f"x{index}" for index in range(5))
    scale = 10**55
    radius = Fraction(1, scale)
    roots = tuple(Fraction(scale + 2 * index + 1, scale) for index in range(5))
    components: tuple[Mapping[tuple[int, ...], int | Fraction], ...] = tuple(
        {
            tuple(int(axis == output) for axis in range(5)): 1,
            (0, 0, 0, 0, 0): -root,
        }
        for output, root in enumerate(roots)
    )

    result = _certify(
        _map(variables, components),
        _box(
            variables,
            tuple((root - radius, root + radius) for root in roots),
        ),
    )

    assert isinstance(result.conclusion, RootBoxCertifiedUniqueNonsingular)
    evidence = result.conclusion.evidence
    assert evidence.center.values == tuple(_q(root) for root in roots)
    assert evidence.value_at_center == (_q(0),) * 5
    assert evidence.krawczyk_image.intervals == tuple(
        _interval(root, root) for root in roots
    )


def test_permuting_the_complete_axis_preserves_the_geometric_root() -> None:
    source = _certify(
        _map(
            ("x", "y"),
            (
                {(1, 0): 1, (0, 1): 1, (0, 0): -3},
                {(1, 0): 1, (0, 1): -1, (0, 0): -1},
            ),
        ),
        _box(
            ("x", "y"),
            ((Fraction(3, 2), Fraction(5, 2)), (Fraction(1, 2), Fraction(3, 2))),
        ),
    )
    transported = _certify(
        _map(
            ("y", "x"),
            (
                {(1, 0): 1, (0, 1): 1, (0, 0): -3},
                {(1, 0): -1, (0, 1): 1, (0, 0): -1},
            ),
        ),
        _box(
            ("y", "x"),
            ((Fraction(1, 2), Fraction(3, 2)), (Fraction(3, 2), Fraction(5, 2))),
        ),
    )

    assert isinstance(source.conclusion, RootBoxCertifiedUniqueNonsingular)
    assert isinstance(transported.conclusion, RootBoxCertifiedUniqueNonsingular)
    assert tuple(
        interval.lower.as_fraction()
        for interval in source.conclusion.evidence.krawczyk_image.intervals
    ) == (Fraction(2), Fraction(1))
    assert tuple(
        interval.lower.as_fraction()
        for interval in transported.conclusion.evidence.krawczyk_image.intervals
    ) == (Fraction(1), Fraction(2))


def test_non_square_system_and_mismatched_axis_are_domain_rejections() -> None:
    x = _polynomial(("x",), {(1,): 1})
    nonsquare = RationalPolynomialMap(input_variables=("x",), output_polynomials=(x, x))
    with pytest.raises(
        OperationDomainValidationError, match="requires a square system"
    ):
        certify_real_root_box(nonsquare, _box(("x",), ((-1, 1),)))

    square = RationalPolynomialMap(input_variables=("x",), output_polynomials=(x,))
    with pytest.raises(OperationDomainValidationError, match="complete ordered axis"):
        certify_real_root_box(square, _box(("y",), ((-1, 1),)))


def test_dimension_and_component_term_boundaries_are_rejected() -> None:
    variables = tuple(f"x{index}" for index in range(MAX_ROOT_BOX_DIMENSION + 1))
    coordinate_components = tuple(
        {tuple(int(axis == output) for axis in range(len(variables))): 1}
        for output in range(len(variables))
    )
    with pytest.raises(OperationDomainValidationError, match="dimension 5"):
        certify_real_root_box(
            _map(variables, coordinate_components),
            _box(variables, tuple((-1, 1) for _ in variables)),
        )

    overfull = _map(
        ("x",),
        ({(degree,): 1 for degree in range(65)},),
    )
    with pytest.raises(OperationDomainValidationError, match="64-term budget"):
        certify_real_root_box(overfull, _box(("x",), ((-1, 1),)))


def test_endpoint_digit_budget_is_owned_by_operation_admission() -> None:
    endpoint = Fraction(10**128, 1)
    with pytest.raises(OperationDomainValidationError, match="128-digit bound"):
        certify_real_root_box(
            _map(("x",), ({(1,): 1},)),
            _box(("x",), ((endpoint, endpoint + 1),)),
        )


@pytest.mark.parametrize("mutation", ("polynomial", "box", "preconditioner"))
def test_record_digest_rejects_independent_source_or_evidence_mutation(
    mutation: str,
) -> None:
    result = _certify(
        _map(("x",), ({(2,): 1, (0,): -2},)),
        _box(("x",), ((1, 2),)),
    )
    payload = result.model_dump(mode="json")
    if mutation == "polynomial":
        payload["polynomial_map"]["output_polynomials"][0]["polynomial"]["terms"][1][
            "coefficient"
        ] = {"num": "-3", "den": "1"}
    elif mutation == "box":
        payload["box"]["intervals"][0]["upper"] = {"num": "3", "den": "1"}
    else:
        payload["conclusion"]["evidence"]["preconditioner"]["entries"][0][0] = {
            "num": "1",
            "den": "2",
        }

    claim = PolynomialSystemRootBoxResult.model_validate(payload)
    assert not verify_real_root_box(claim)


def test_produced_result_round_trips_and_schema_discriminates_every_branch() -> None:
    result = _certify(
        _map(("x",), ({(2,): 1, (0,): -2},)),
        _box(("x",), ((1, 2),)),
    )

    assert (
        PolynomialSystemRootBoxResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    schema = json.dumps(PolynomialSystemRootBoxResult.model_json_schema())
    assert '"discriminator": {"mapping"' in schema
    assert '"propertyName": "status"' in schema
    assert '"propertyName": "method"' in schema
    assert '"propertyName": "kind"' in schema


def test_conclusion_union_rejects_cross_branch_field_combinations() -> None:
    certified = _certify(
        _map(("x",), ({(2,): 1, (0,): -2},)),
        _box(("x",), ((1, 2),)),
    ).model_dump(mode="json")
    certified["conclusion"]["status"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        PolynomialSystemRootBoxResult.model_validate(certified)

    unknown = _certify(
        _map(("x",), ({(2,): 1},)),
        _box(("x",), ((-1, 1),)),
    ).model_dump(mode="json")
    unknown["conclusion"]["status"] = "CERTIFIED_UNIQUE_NONSINGULAR"
    with pytest.raises(ValidationError):
        PolynomialSystemRootBoxResult.model_validate(unknown)


def test_public_declaration_has_one_executable_square_system_example() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert tool.operation_id == "polynomial.system.real_root_box.certify"
    assert tool.request_type is PolynomialSystemRootBoxRequest
    assert tool.result_type is PolynomialSystemRootBoxResult
    assert len(tool.examples) == 1
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert isinstance(result.conclusion, RootBoxCertifiedUniqueNonsingular)


def test_linear_system_krawczyk_image_is_the_exact_known_root() -> None:
    polynomial_map = _map(
        ("x", "y"),
        (
            {(1, 0): 1, (0, 1): 1, (0, 0): -3},
            {(1, 0): 1, (0, 1): -1, (0, 0): -1},
        ),
    )
    box = _box(
        ("x", "y"),
        (
            (Fraction(3, 2), Fraction(5, 2)),
            (Fraction(1, 2), Fraction(3, 2)),
        ),
    )
    result = _certify(polynomial_map, box)

    assert isinstance(result.conclusion, RootBoxCertifiedUniqueNonsingular)
    image = result.conclusion.evidence.krawczyk_image
    assert image.intervals == (_interval(2, 2), _interval(1, 1))
    assert tuple(interval.lower.as_fraction() for interval in image.intervals) == (
        Fraction(2),
        Fraction(1),
    )
