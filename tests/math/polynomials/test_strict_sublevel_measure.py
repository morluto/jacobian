"""Exact contract and invariant tests for strict polynomial sublevel measure."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.real_algebra._operations import (
    compute_strict_sublevel_measure,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    LevelRootEndpoint,
    ScopeEndpoint,
    StrictSublevelMeasureRequest,
    StrictSublevelMeasureResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def _polynomial(
    *terms: tuple[int | Fraction, int],
    variables: tuple[str, ...] = ("x",),
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=(exponent,) + (0,) * (len(variables) - 1),
                )
                for coefficient, exponent in sorted(
                    terms,
                    key=lambda item: item[1],
                    reverse=True,
                )
                if coefficient
            )
        ),
    )


def _request(
    polynomial: RationalPolynomial,
    *,
    threshold: int = 1,
    lower: int = -2,
    upper: int = 2,
) -> StrictSublevelMeasureRequest:
    return StrictSublevelMeasureRequest(
        polynomial=polynomial,
        threshold=_rational(threshold),
        lower=_rational(lower),
        upper=_rational(upper),
    )


def _resolve_endpoint(result: StrictSublevelMeasureResult, endpoint: object) -> Any:
    from sympy import Rational

    if isinstance(endpoint, ScopeEndpoint):
        return Rational(*endpoint.value.as_integer_ratio())
    assert isinstance(endpoint, LevelRootEndpoint)
    polynomial = rational_polynomial_to_sympy(result.source_polynomial)
    threshold = Rational(*result.threshold.as_integer_ratio())
    level = (
        polynomial - threshold
        if endpoint.root.equation == "F_MINUS_THRESHOLD"
        else polynomial + threshold
    )
    return level.real_roots(multiple=False, radicals=False)[endpoint.root.root_index][0]


def _resolve_measure(result: StrictSublevelMeasureResult) -> Any:
    from sympy import Rational

    polynomial = rational_polynomial_to_sympy(result.source_polynomial)
    threshold = Rational(*result.threshold.as_integer_ratio())
    value = Rational(*result.measure.rational_part.as_integer_ratio())
    for term in result.measure.root_terms:
        level = (
            polynomial - threshold
            if term.root.equation == "F_MINUS_THRESHOLD"
            else polynomial + threshold
        )
        root = level.real_roots(multiple=False, radicals=False)[term.root.root_index][0]
        value += term.coefficient * root
    return value


def test_quadratic_measure_retains_exact_irrational_boundary_sum() -> None:
    result = compute_strict_sublevel_measure(_request(_polynomial((1, 2)), threshold=2))

    assert len(result.components) == 1
    component = result.components[0]
    assert isinstance(component.left, LevelRootEndpoint)
    assert isinstance(component.right, LevelRootEndpoint)
    assert component.left.root.equation == "F_MINUS_THRESHOLD"
    assert component.left.root.root_index == 0
    assert component.right.root.root_index == 1
    assert not component.left_included
    assert not component.right_included
    assert result.measure.rational_part == _rational(0)
    assert tuple(term.coefficient for term in result.measure.root_terms) == (-1, 1)

    reconstructed_length = sum(
        _resolve_endpoint(result, item.right) - _resolve_endpoint(result, item.left)
        for item in result.components
    )
    assert _resolve_measure(result) - reconstructed_length == 0


def test_producer_isolates_once_and_external_validation_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.real_algebra._operations as operations
    import jacobian.math.polynomials.real_algebra._strict_sublevel as kernel

    calls = 0
    original = kernel.compute_strict_sublevel_payload

    def counting(request: StrictSublevelMeasureRequest) -> Any:
        nonlocal calls
        calls += 1
        return original(request)

    monkeypatch.setattr(operations, "compute_strict_sublevel_payload", counting)
    monkeypatch.setattr(kernel, "compute_strict_sublevel_payload", counting)
    result = operations.compute_strict_sublevel_measure(
        _request(_polynomial((1, 2)), threshold=2)
    )
    assert calls == 1

    StrictSublevelMeasureResult.model_validate(result.model_dump(mode="json"))
    assert calls == 2


def test_request_and_result_compose_through_strict_json_transport() -> None:
    request = _request(_polynomial((1, 2)), threshold=2)
    parsed_request = StrictSublevelMeasureRequest.model_validate_json(
        request.model_dump_json(), strict=True
    )
    result = compute_strict_sublevel_measure(parsed_request)

    assert parsed_request == request
    assert (
        StrictSublevelMeasureResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )


def test_rational_level_roots_collapse_into_the_rational_measure_part() -> None:
    result = compute_strict_sublevel_measure(
        StrictSublevelMeasureRequest(
            polynomial=_polynomial((Fraction(1, 2), 1)),
            threshold=_rational(1, 2),
            lower=_rational(-3, 2),
            upper=_rational(3, 2),
        )
    )

    assert len(result.components) == 1
    component = result.components[0]
    assert isinstance(component.left, LevelRootEndpoint)
    assert component.left.root.equation == "F_PLUS_THRESHOLD"
    assert isinstance(component.right, LevelRootEndpoint)
    assert component.right.root.equation == "F_MINUS_THRESHOLD"
    assert result.measure.rational_part == _rational(2)
    assert result.measure.root_terms == ()


@pytest.mark.parametrize(
    ("polynomial", "threshold", "component_count", "measure"),
    [
        (_polynomial(), 1, 1, 4),
        (_polynomial(), 0, 0, 0),
        (_polynomial((1, 0)), 2, 1, 4),
        (_polynomial((2, 0)), 2, 0, 0),
        (_polynomial((3, 0)), 2, 0, 0),
    ],
)
def test_zero_and_constant_polynomial_conventions(
    polynomial: RationalPolynomial,
    threshold: int,
    component_count: int,
    measure: int,
) -> None:
    result = compute_strict_sublevel_measure(_request(polynomial, threshold=threshold))

    assert len(result.components) == component_count
    assert result.measure.rational_part == _rational(measure)
    assert result.measure.root_terms == ()
    if result.components:
        component = result.components[0]
        assert isinstance(component.left, ScopeEndpoint)
        assert isinstance(component.right, ScopeEndpoint)
        assert component.left_included and component.right_included


def test_even_root_splits_strict_components_without_sign_change() -> None:
    # |x^2-1| < 1 is (-sqrt(2), 0) union (0, sqrt(2)).  The root x=0 of
    # f+1 has multiplicity two: the product sign stays negative, but strict
    # equality removes the point and therefore separates the components.
    result = compute_strict_sublevel_measure(_request(_polynomial((1, 2), (-1, 0))))

    assert len(result.components) == 2
    tangent_right = result.components[0].right
    tangent_left = result.components[1].left
    assert tangent_right == tangent_left
    assert isinstance(tangent_right, LevelRootEndpoint)
    assert tangent_right.root.equation == "F_PLUS_THRESHOLD"
    assert tangent_right.root.multiplicity == 2
    assert not result.components[0].right_included
    assert not result.components[1].left_included
    assert all(
        term.root.equation == "F_MINUS_THRESHOLD" for term in result.measure.root_terms
    )


def test_roots_at_scope_endpoints_remain_excluded_under_strict_semantics() -> None:
    result = compute_strict_sublevel_measure(
        _request(_polynomial((1, 1)), lower=-1, upper=1)
    )

    assert len(result.components) == 1
    component = result.components[0]
    assert isinstance(component.left, ScopeEndpoint)
    assert isinstance(component.right, ScopeEndpoint)
    assert not component.left_included
    assert not component.right_included
    assert result.measure.rational_part == _rational(2)


def test_singleton_scope_preserves_membership_while_measure_stays_zero() -> None:
    inside = compute_strict_sublevel_measure(
        _request(_polynomial((1, 1)), lower=0, upper=0)
    )
    equality = compute_strict_sublevel_measure(
        _request(_polynomial((1, 1)), lower=1, upper=1)
    )

    assert len(inside.components) == 1
    assert inside.components[0].left == inside.components[0].right
    assert inside.components[0].left_included
    assert inside.components[0].right_included
    assert inside.measure.rational_part == _rational(0)
    assert equality.components == ()
    assert equality.measure.rational_part == _rational(0)


def test_request_rejects_outside_domain_and_work_bounds() -> None:
    with pytest.raises(ValidationError, match="one polynomial variable"):
        _request(_polynomial((1, 1), variables=("x", "y")))
    with pytest.raises(ValidationError, match="nonnegative"):
        StrictSublevelMeasureRequest(
            polynomial=_polynomial((1, 1)),
            threshold=_rational(-1),
            lower=_rational(-1),
            upper=_rational(1),
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        _request(_polynomial((1, 1)), lower=2, upper=-2)
    with pytest.raises(ValidationError, match="16-degree operation budget"):
        _request(_polynomial((1, 17)))
    _request(_polynomial((1, 16)))

    oversized_coefficient = Fraction(int("9" * 65), 1)
    with pytest.raises(ValidationError, match="64-digit bound"):
        _request(_polynomial((oversized_coefficient, 1)))


def test_request_bounds_raw_polynomial_before_nested_parsing() -> None:
    raw_term = {
        "coefficient": {"num": "not-an-integer", "den": "1"},
        "exponents": [0],
    }
    with pytest.raises(ValidationError, match="17-term operation budget"):
        StrictSublevelMeasureRequest.model_validate(
            {
                "polynomial": {
                    "variables": ["x"],
                    "polynomial": {"terms": [raw_term] * 18},
                },
                "threshold": {"num": "1", "den": "1"},
                "lower": {"num": "-2", "den": "1"},
                "upper": {"num": "2", "den": "1"},
            }
        )

    with pytest.raises(ValidationError, match="64-digit bound"):
        StrictSublevelMeasureRequest.model_validate(
            {
                "polynomial": {
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "x" * 65, "den": "1"},
                                "exponents": [1],
                            }
                        ]
                    },
                },
                "threshold": {"num": "1", "den": "1"},
                "lower": {"num": "-2", "den": "1"},
                "upper": {"num": "2", "den": "1"},
            }
        )

    with pytest.raises(ValidationError, match="exactly one exponent per term"):
        StrictSublevelMeasureRequest.model_validate(
            {
                "polynomial": {
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "not-an-integer", "den": "1"},
                                "exponents": [0] * 65,
                            }
                        ]
                    },
                },
                "threshold": {"num": "1", "den": "1"},
                "lower": {"num": "-2", "den": "1"},
                "upper": {"num": "2", "den": "1"},
            }
        )


def test_request_preflights_cleared_level_polynomial_height() -> None:
    pairwise_coprime_denominators = (
        2**210,
        3**130,
        5**90,
        7**74,
        11**60,
    )
    polynomial = _polynomial(
        *(
            (Fraction(1, denominator), exponent)
            for exponent, denominator in enumerate(pairwise_coprime_denominators)
        )
    )

    with pytest.raises(ValidationError, match="root-isolation bound"):
        _request(polynomial)

    # No root work is needed for the strict t=0 set, so result-sensitive
    # admission accepts the same source and returns the exact empty set.
    request = _request(polynomial, threshold=0)
    assert compute_strict_sublevel_measure(request).components == ()


def test_request_jointly_bounds_level_root_isolation_degree_and_height() -> None:
    scale = 10**20
    close_root_polynomial = _polynomial(
        (1, 8),
        (-2 * scale**2, 2),
        (4 * scale, 1),
        (-1, 0),
    )

    with pytest.raises(ValidationError, match="exact-root isolation exceeds"):
        _request(close_root_polynomial)


def test_small_integer_polynomials_match_sympy_inequality_sets_exhaustively() -> None:
    """Differentially compare every degree-one/two polynomial in {-1,0,1}[x]."""

    from sympy import Abs, Interval, S, Symbol, Union, simplify
    from sympy.solvers.inequalities import solve_univariate_inequality

    x = Symbol("x", real=True)
    for degree in (1, 2):
        for lower_coefficients in product((-1, 0, 1), repeat=degree):
            for leading_coefficient in (-1, 1):
                coefficients = (leading_coefficient, *lower_coefficients)
                polynomial = _polynomial(
                    *(
                        (coefficient, degree - index)
                        for index, coefficient in enumerate(coefficients)
                    )
                )
                result = compute_strict_sublevel_measure(_request(polynomial))
                actual_intervals = tuple(
                    Interval(
                        _resolve_endpoint(result, component.left),
                        _resolve_endpoint(result, component.right),
                        left_open=not component.left_included,
                        right_open=not component.right_included,
                    )
                    for component in result.components
                )

                expression = sum(
                    coefficient * x ** (degree - index)
                    for index, coefficient in enumerate(coefficients)
                )
                expected_set = solve_univariate_inequality(
                    Abs(expression) < 1,
                    x,
                    relational=False,
                ).intersect(Interval(-2, 2))
                if expected_set is S.EmptySet:
                    expected_intervals: tuple[Interval, ...] = ()
                elif isinstance(expected_set, Interval):
                    expected_intervals = (expected_set,)
                else:
                    assert isinstance(expected_set, Union)
                    expected_intervals = tuple(expected_set.args)

                assert len(actual_intervals) == len(expected_intervals), expression
                for actual, expected in zip(
                    actual_intervals,
                    expected_intervals,
                    strict=True,
                ):
                    assert actual.left_open == expected.left_open, expression
                    assert actual.right_open == expected.right_open, expression
                    assert actual.start.equals(expected.start), expression
                    assert actual.end.equals(expected.end), expression
                assert simplify(_resolve_measure(result) - expected_set.measure) == 0


def test_source_bound_result_rejects_independent_forged_fields() -> None:
    result = compute_strict_sublevel_measure(_request(_polynomial((1, 2)), threshold=2))
    payload = result.model_dump(mode="json")
    mutations = []

    changed_source = deepcopy(payload)
    changed_source["source_polynomial"]["polynomial"]["terms"].append(
        {
            "coefficient": {"num": "3", "den": "1"},
            "exponents": (0,),
        }
    )
    mutations.append(changed_source)

    changed_threshold = deepcopy(payload)
    changed_threshold["threshold"] = {"num": "0", "den": "1"}
    mutations.append(changed_threshold)

    changed_scope = deepcopy(payload)
    changed_scope["upper"] = {"num": "0", "den": "1"}
    mutations.append(changed_scope)

    changed_endpoint = deepcopy(payload)
    changed_endpoint["components"][0]["left"]["root"]["root_index"] = 1
    mutations.append(changed_endpoint)

    changed_measure = deepcopy(payload)
    changed_measure["measure"]["root_terms"][0]["coefficient"] = 1
    mutations.append(changed_measure)

    for mutation in mutations:
        with pytest.raises(ValidationError):
            StrictSublevelMeasureResult.model_validate(mutation)


def test_measure_root_incidence_rejects_boolean_coefficient() -> None:
    result = compute_strict_sublevel_measure(_request(_polynomial((1, 2)), threshold=2))
    payload = result.model_dump(mode="json")
    payload["measure"]["root_terms"][0]["coefficient"] = True

    with pytest.raises(ValidationError, match="integer -1 or 1"):
        StrictSublevelMeasureResult.model_validate(payload)


def test_result_bounds_raw_measure_before_source_replay() -> None:
    result = compute_strict_sublevel_measure(_request(_polynomial((1, 2)), threshold=2))
    payload = result.model_dump(mode="json")
    payload["measure"]["rational_part"] = {
        "num": "x" * 8_323,
        "den": "1",
    }

    with pytest.raises(ValidationError, match="8322-digit bound"):
        StrictSublevelMeasureResult.model_validate(payload)
