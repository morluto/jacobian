"""Exact contract tests for certified unit-circle supremum norms.

The operation returns a certified rational enclosure of ``max_{|z|=1} |P(z)|^2``
together with the complete critical-point ledger, plus the exact maximum as
an indexed real-algebraic root whenever its irreducible resultant factor fits
the shared carrier; these tests check the known closed-form values, the
independent defining identity ``|P(z)|^2 = Q(t) / (1 + t**2)**d``, the
``z = -1`` endpoint, the full-circle degenerate branch, exact-value replay
and forgery rejection, and the admission envelope.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.polynomials.unit_circle._sup_norm import (
    unit_circle_sup_norm_squared,
    verify_unit_circle_sup_norm_squared,
)
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    MAX_SUP_NORM_DEGREE,
    UnitCircleSupNormSquaredRequest,
    UnitCircleSupNormSquaredResult,
)
from jacobian.math.polynomials.unit_circle._tools import TOOLS


def _payload(
    coefficients: dict[int, tuple[int, int]],
) -> dict[str, object]:
    terms = [
        {
            "coefficient": {
                "real": {"num": str(real), "den": "1"},
                "imaginary": {"num": str(imag), "den": "1"},
            },
            "exponent": exponent,
        }
        for exponent, (real, imag) in sorted(coefficients.items(), reverse=True)
    ]
    return {
        "polynomial": {"domain": "QQ(i)", "variable": "z", "terms": terms},
    }


def _result(coefficients: dict[int, tuple[int, int]]) -> UnitCircleSupNormSquaredResult:
    request = UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps(_payload(coefficients))
    )
    return unit_circle_sup_norm_squared(request.polynomial)


def _exact(result: UnitCircleSupNormSquaredResult) -> Fraction | None:
    """Return the enclosure as one rational when it is a singleton."""

    enclosure = result.sup_norm_squared_enclosure
    if enclosure.lower == enclosure.upper:
        return Fraction(enclosure.lower.as_integer_ratio()[0], enclosure.lower.den)
    return None


def _gaussian(real: int, imaginary: int) -> GaussianRational:
    return GaussianRational.model_validate_json(
        json.dumps(
            {
                "real": {"num": str(real), "den": "1"},
                "imaginary": {"num": str(imaginary), "den": "1"},
            }
        )
    )


def _evaluate_transformed(
    coefficients_ascending: tuple[Fraction, ...], t: Fraction
) -> Fraction:
    total = Fraction(0)
    for coefficient in reversed(coefficients_ascending):
        total = total * t + coefficient
    return total


def _horner_descending(descending: tuple[int, ...], value: Fraction) -> Fraction:
    total = Fraction(0)
    for coefficient in descending:
        total = total * value + coefficient
    return total


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ({0: (1, 0)}, Fraction(1)),
        ({1: (1, 0)}, Fraction(1)),
        ({2: (1, 0)}, Fraction(1)),
        ({0: (1, 0), 1: (1, 0)}, Fraction(4)),
        ({0: (1, 0), 1: (-1, 0)}, Fraction(4)),
        ({0: (-1, 0), 2: (1, 0)}, Fraction(4)),
        ({0: (1, 0), 1: (0, 1)}, Fraction(4)),
    ],
)
def test_known_sup_norm_squared_values(
    coefficients: dict[int, tuple[int, int]], expected: Fraction
) -> None:
    result = _result(coefficients)

    assert _exact(result) == expected
    assert (
        Fraction(*result.sup_norm_enclosure.lower.as_integer_ratio()) ** 2 <= expected
    )
    assert (
        Fraction(*result.sup_norm_enclosure.upper.as_integer_ratio()) ** 2 >= expected
    )


def test_zero_polynomial_has_zero_sup_norm() -> None:
    result = _result({})

    assert _exact(result) == 0
    assert result.degree == 0
    assert result.critical_points == ()


def test_constant_modulus_is_a_full_circle_maximizer_set() -> None:
    for coefficients in ({0: (1, 0)}, {1: (1, 0)}, {2: (1, 0)}, {}):
        result = _result(coefficients)
        assert result.maximizing_status == "FULL_CIRCLE"


def test_endpoint_minus_one_is_the_unique_maximizer_for_one_minus_z() -> None:
    result = _result({0: (1, 0), 1: (-1, 0)})

    assert result.maximizing_status == "ENDPOINT"
    assert result.endpoint_minus_one_is_maximizer
    assert _exact(result) == 4
    assert all(not point.is_maximizer for point in result.critical_points)


def test_isolated_critical_maximizer_is_reported_with_zero_rank() -> None:
    result = _result({0: (1, 0), 1: (1, 0)})

    assert result.maximizing_status == "ISOLATED_CRITICAL"
    maximizers = [point for point in result.critical_points if point.is_maximizer]
    assert len(maximizers) == 1
    assert maximizers[0].comparison_rank == 0
    assert all(not point.is_maximizer for point in result.critical_points[1:])


def test_defining_identity_replays_the_transformed_rational_function() -> None:
    coefficients = {0: (1, 0), 1: (1, 0)}
    result = _result(coefficients)

    ascending = tuple(
        Fraction(*value.as_integer_ratio())
        for value in reversed(result.transformed_numerator)
    )
    exponent = result.denominator_exponent
    for point in result.critical_points:
        if point.parameter.interval_type != "SINGLETON":
            continue
        t = Fraction(*point.parameter.lower.as_integer_ratio())
        value = _evaluate_transformed(ascending, t) / (1 + t * t) ** exponent
        assert Fraction(*point.value.lower.as_integer_ratio()) <= value
        assert value <= Fraction(*point.value.upper.as_integer_ratio())


def test_critical_ledger_brackets_a_dense_grid_lower_bound() -> None:
    # Degree-three Littlewood polynomial: the true maximum is an algebraic
    # number, so the operation must return a non-singleton certified interval
    # that dominates every sampled value.
    coefficients = {0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)}
    result = _result(coefficients)

    lower = Fraction(*result.sup_norm_squared_enclosure.lower.as_integer_ratio())
    upper = Fraction(*result.sup_norm_squared_enclosure.upper.as_integer_ratio())
    assert lower < upper

    grid_max = 0.0
    for step in range(20_000):
        theta = 2 * math.pi * step / 20_000
        modulus = abs(
            sum(
                complex(real, imag)
                * complex(math.cos(theta), math.sin(theta)) ** exponent
                for exponent, (real, imag) in coefficients.items()
            )
        )
        grid_max = max(grid_max, modulus**2)
    assert grid_max <= float(lower) + 1e-9
    assert float(lower) <= float(upper)

    derivative_roots = len(result.critical_points)
    assert 0 < derivative_roots <= 2 * result.degree + 1


def test_enclosure_is_consistent_with_the_square_root_certificate() -> None:
    result = _result({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)})

    root_lower = Fraction(*result.sup_norm_enclosure.lower.as_integer_ratio())
    root_upper = Fraction(*result.sup_norm_enclosure.upper.as_integer_ratio())
    square_lower = Fraction(*result.sup_norm_squared_enclosure.lower.as_integer_ratio())
    square_upper = Fraction(*result.sup_norm_squared_enclosure.upper.as_integer_ratio())

    assert root_lower >= 0
    assert root_lower**2 <= square_lower
    assert root_upper**2 >= square_upper


def test_native_and_catalog_paths_agree() -> None:
    request = UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps(_payload({0: (1, 0), 1: (1, 0)}))
    )
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.unit_circle.sup_norm_squared.compute"
    )

    assert tool.run(request) == unit_circle_sup_norm_squared(request.polynomial)


def test_result_and_claim_round_trip_through_serialization() -> None:
    result = _result({0: (1, 0), 1: (-1, 0)})
    decoded = UnitCircleSupNormSquaredResult.model_validate_json(
        result.model_dump_json()
    )

    assert decoded == result
    assert decoded.maximizing_status == "ENDPOINT"


def test_declaration_example_executes() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.unit_circle.sup_norm_squared.compute"
    )

    example_request = UnitCircleSupNormSquaredRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(example_request)

    assert isinstance(result, UnitCircleSupNormSquaredResult)
    assert _exact(result) == 4


def test_degree_above_ceiling_is_rejected() -> None:
    coefficients = dict.fromkeys(range(MAX_SUP_NORM_DEGREE + 2), (1, 0))

    with pytest.raises(ValidationError):
        _result(coefficients)


def test_native_boundary_rejects_a_forged_degree_above_the_envelope() -> None:
    from jacobian.math.polynomials.unit_circle._sup_norm_models import (
        GaussianRationalPolynomial,
        GaussianRationalPolynomialTerm,
    )

    forged = GaussianRationalPolynomial.model_construct(
        domain="QQ(i)",
        variable="z",
        terms=(
            GaussianRationalPolynomialTerm.model_construct(
                coefficient=_gaussian(1, 0),
                exponent=MAX_SUP_NORM_DEGREE + 1,
            ),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError):
        unit_circle_sup_norm_squared(forged)


def test_duplicate_exponents_are_rejected() -> None:
    payload = _payload({1: (1, 0), 0: (1, 0)})
    payload["polynomial"]["terms"] = [  # type: ignore[index]
        {
            "coefficient": {
                "real": {"num": "1", "den": "1"},
                "imaginary": {"num": "0", "den": "1"},
            },
            "exponent": 1,
        },
        {
            "coefficient": {
                "real": {"num": "1", "den": "1"},
                "imaginary": {"num": "0", "den": "1"},
            },
            "exponent": 1,
        },
    ]

    with pytest.raises(ValidationError):
        UnitCircleSupNormSquaredRequest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ({0: (1, 0), 1: (1, 0), 2: (1, 0)}, (1, -9)),
        ({1: (1, 1)}, (1, -2)),
        ({0: (1, 0), 1: (-1, 0)}, (1, -4)),
        ({0: (-1, 0), 2: (-2, 0), 3: (1, 0)}, (1, -16)),
    ],
)
def test_exact_maximum_for_rational_maximizers(
    coefficients: dict[int, tuple[int, int]], expected: tuple[int, int]
) -> None:
    result = _result(coefficients)

    assert result.sup_norm_squared_exact is not None
    assert tuple(result.sup_norm_squared_exact.polynomial) == expected
    assert result.sup_norm_squared_exact.real_root_index == 0
    assert verify_unit_circle_sup_norm_squared(result)


@pytest.mark.parametrize(
    ("coefficients", "expected", "index"),
    [
        ({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)}, (27, -216, 176), 1),
        (
            {0: (-2, 0), 2: (2, 0), 3: (-3, 0), 4: (-3, 0)},
            (4608, -318684, 2623484, 8833),
            2,
        ),
    ],
)
def test_exact_maximum_for_irrational_maximizers(
    coefficients: dict[int, tuple[int, int]],
    expected: tuple[int, ...],
    index: int,
) -> None:
    result = _result(coefficients)

    exact = result.sup_norm_squared_exact
    assert exact is not None
    assert tuple(exact.polynomial) == expected
    assert exact.real_root_index == index
    # Independent replay without the SymPy backend: the retained enclosure
    # strictly brackets one root of the claimed minimal polynomial.
    enclosure = result.sup_norm_squared_enclosure
    lower = Fraction(*enclosure.lower.as_integer_ratio())
    upper = Fraction(*enclosure.upper.as_integer_ratio())
    assert lower < upper
    assert _horner_descending(expected, lower) * _horner_descending(expected, upper) < 0
    assert verify_unit_circle_sup_norm_squared(result)


def test_exact_maximum_round_trip_through_serialization() -> None:
    result = _result({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)})
    decoded = UnitCircleSupNormSquaredResult.model_validate_json(
        result.model_dump_json()
    )

    assert decoded == result
    assert decoded.sup_norm_squared_exact is not None
    assert tuple(decoded.sup_norm_squared_exact.polynomial) == (27, -216, 176)


def test_forged_exact_maximum_is_rejected() -> None:
    result = _result({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)})
    forged = result.model_copy(
        update={
            "sup_norm_squared_exact": RealAlgebraicValue._from_admitted_polynomial(
                polynomial=(1, -999),
                real_root_index=0,
            )
        }
    )

    assert not verify_unit_circle_sup_norm_squared(forged)


def test_weakened_exact_maximum_is_rejected() -> None:
    result = _result({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)})
    weakened = result.model_copy(update={"sup_norm_squared_exact": None})

    assert not verify_unit_circle_sup_norm_squared(weakened)


def test_forged_aggregate_fields_are_rejected() -> None:
    result = _result({0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)})
    enclosure = result.sup_norm_squared_enclosure

    forged_enclosure = result.model_copy(
        update={
            "sup_norm_squared_enclosure": enclosure.model_copy(
                update={
                    "lower": CanonicalRational(num=1, den=1),
                    "upper": CanonicalRational(num=1_000_000, den=1),
                }
            )
        }
    )
    forged_rank = result.model_copy(
        update={
            "critical_points": tuple(
                point.model_copy(update={"comparison_rank": 99, "is_maximizer": True})
                for point in result.critical_points
            )
        }
    )
    forged_status = result.model_copy(update={"maximizing_status": "ENDPOINT"})

    assert not verify_unit_circle_sup_norm_squared(forged_enclosure)
    assert not verify_unit_circle_sup_norm_squared(forged_rank)
    assert not verify_unit_circle_sup_norm_squared(forged_status)


def test_top_admitted_degree_carries_an_exact_value() -> None:
    # Degree-8 input: the leading t**(2m+1) terms of Q'(1+t^2) and 2*d*t*Q
    # cancel, so every critical-value minimal factor has degree at most 2*d
    # and fits the shared carrier.  This pins that bound at the envelope.
    result = _result(
        {exponent: (1 if exponent % 3 else -1, 0) for exponent in range(9)}
    )

    assert result.degree == MAX_SUP_NORM_DEGREE
    exact = result.sup_norm_squared_exact
    assert exact is not None
    assert tuple(exact.polynomial) == (
        65536,
        -3474944,
        43543305,
        -179214444,
        225009495,
    )
    assert exact.real_root_index == 3
    assert len(exact.polynomial) - 1 <= 2 * result.degree
    assert verify_unit_circle_sup_norm_squared(result)


@pytest.mark.parametrize(
    "coefficients",
    [
        {0: (1, 0), 1: (1, 0), 2: (-1, 0), 3: (1, 0)},
        {0: (-2, 0), 2: (2, 0), 3: (-3, 0), 4: (-3, 0)},
        {exponent: (1 if exponent % 3 else -1, 0) for exponent in range(9)},
    ],
)
def test_exact_value_respects_the_resultant_degree_carrier(
    coefficients: dict[int, tuple[int, int]],
) -> None:
    result = _result(coefficients)

    exact = result.sup_norm_squared_exact
    if exact is None:
        # Only the theoretical height overflow may omit the exact value.
        return
    assert len(exact.polynomial) - 1 <= 2 * result.degree
    assert exact.real_root_index < len(exact.polynomial) - 1
    assert verify_unit_circle_sup_norm_squared(result)
