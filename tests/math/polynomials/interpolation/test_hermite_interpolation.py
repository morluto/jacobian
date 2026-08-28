"""Exact contract, replay, and boundary tests for Hermite interpolation."""

from __future__ import annotations

from fractions import Fraction
from math import prod

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials import interpolation
from jacobian.math.polynomials._elementary_operations import (
    rational_polynomial_evaluate,
)
from jacobian.math.polynomials._models import RationalPolynomialEvaluationRequest
from jacobian.math.polynomials.interpolation import (
    HermiteInterpolationResult,
    OrdinaryDerivativeJet,
    OrdinaryDerivativeJetTable,
    OrdinaryDerivativeValue,
    hermite_interpolation,
)
from jacobian.math.polynomials.interpolation._models import (
    _MAX_RATIONAL_DIGITS,
    MAX_HERMITE_CUBIC_WORK_CELLS,
    MAX_HERMITE_RESULT_BYTES,
    MAX_HERMITE_SYSTEM_CELLS,
    HermiteInterpolationRequest,
)
from jacobian.math.polynomials.interpolation._operations import (
    compute_hermite_interpolation,
)


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    value = Fraction(numerator, denominator)
    return CanonicalRational(
        num=str(value.numerator),
        den=str(value.denominator),
    )


def _derivative(
    derivative_order: int,
    numerator: int,
    denominator: int = 1,
) -> OrdinaryDerivativeValue:
    return OrdinaryDerivativeValue(
        derivative_order=derivative_order,
        value=_q(numerator, denominator),
    )


def _jet(
    node_numerator: int,
    *values: tuple[int, int],
    node_denominator: int = 1,
) -> OrdinaryDerivativeJet:
    return OrdinaryDerivativeJet(
        node=_q(node_numerator, node_denominator),
        derivatives=tuple(
            _derivative(order, numerator, denominator)
            for order, (numerator, denominator) in enumerate(values)
        ),
    )


def _table(*jets: OrdinaryDerivativeJet) -> OrdinaryDerivativeJetTable:
    return OrdinaryDerivativeJetTable(variable="x", jets=jets)


def _full_multiplicity_payload(
    node_digits: int,
    *,
    last_value_digits: int | None = None,
) -> dict[str, object]:
    return {
        "variable": "x",
        "jets": [
            {
                "node": {"num": "9" * node_digits, "den": "1"},
                "derivatives": [
                    {
                        "derivative_order": order,
                        "value": {
                            "num": (
                                "9" * last_value_digits
                                if order == 31 and last_value_digits is not None
                                else "0"
                            ),
                            "den": "1",
                        },
                    }
                    for order in range(32)
                ],
            }
        ],
    }


def _coefficient_map(result: HermiteInterpolationResult) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in result.polynomial.polynomial.terms
    }


def _ordinary_derivative(
    coefficients: tuple[Fraction, ...], node: Fraction, order: int
) -> Fraction:
    return sum(
        (
            coefficients[degree]
            * prod(range(degree - order + 1, degree + 1))
            * node ** (degree - order)
            for degree in range(order, len(coefficients))
        ),
        start=Fraction(0),
    )


def test_native_api_exports_the_typed_hermite_operation() -> None:
    assert tuple(interpolation.__all__) == (
        "HermiteConstraintReplay",
        "HermiteInterpolationResult",
        "OrdinaryDerivativeJet",
        "OrdinaryDerivativeJetTable",
        "OrdinaryDerivativeValue",
        "hermite_interpolation",
    )
    assert interpolation.hermite_interpolation is hermite_interpolation


def test_constant_jet_returns_a_canonical_constant_polynomial() -> None:
    result = hermite_interpolation(_table(_jet(3, (5, 1))))

    assert _coefficient_map(result) == {0: Fraction(5)}
    assert result.total_multiplicity == 1
    assert result.degree == 0
    assert result.leading_coefficient.as_fraction() == 5


def test_higher_jet_uses_ordinary_not_hasse_derivatives() -> None:
    result = hermite_interpolation(_table(_jet(2, (8, 1), (12, 1), (12, 1), (6, 1))))

    assert _coefficient_map(result) == {3: Fraction(1)}
    assert result.degree == 3
    assert tuple(item.computed.as_fraction() for item in result.replay) == (
        Fraction(8),
        Fraction(12),
        Fraction(12),
        Fraction(6),
    )


def test_known_polynomial_is_reconstructed_from_several_nodes() -> None:
    # p(x) = 2*x^5 - 3/2*x^3 + 7*x - 4, with M = 2 + 3 + 1 = 6.
    table = _table(
        _jet(-1, (-23, 2), (25, 2)),
        _jet(0, (-4, 1), (7, 1), (0, 1)),
        _jet(2, (62, 1)),
    )

    result = compute_hermite_interpolation(HermiteInterpolationRequest(table=table))

    assert _coefficient_map(result) == {
        5: Fraction(2),
        3: Fraction(-3, 2),
        1: Fraction(7),
        0: Fraction(-4),
    }
    assert result.degree == 5
    assert result.leading_coefficient.as_fraction() == 2
    assert len(result.replay) == result.total_multiplicity == 6


def test_rational_nodes_follow_the_same_exact_contract() -> None:
    result = hermite_interpolation(
        _table(
            _jet(1, (1, 4), (1, 1), node_denominator=2),
            _jet(-1, (1, 1)),
        )
    )

    assert _coefficient_map(result) == {2: Fraction(1)}
    assert tuple(item.computed.as_fraction() for item in result.replay) == (
        Fraction(1),
        Fraction(1, 4),
        Fraction(1),
    )


def test_replay_establishes_every_defining_derivative_constraint() -> None:
    coefficients = (
        Fraction(-4),
        Fraction(7),
        Fraction(0),
        Fraction(-3, 2),
        Fraction(0),
        Fraction(2),
    )
    table = _table(
        _jet(-1, (-23, 2), (25, 2)),
        _jet(0, (-4, 1), (7, 1), (0, 1)),
        _jet(2, (62, 1)),
    )
    result = hermite_interpolation(table)

    for replay in result.replay:
        assert replay.computed.as_fraction() == _ordinary_derivative(
            coefficients,
            replay.node.as_fraction(),
            replay.derivative_order,
        )
        assert replay.computed.as_fraction() == replay.expected.as_fraction()


def test_zero_polynomial_retains_ring_and_zero_conventions() -> None:
    result = hermite_interpolation(
        _table(
            _jet(-2, (0, 1), (0, 1)),
            _jet(3, (0, 1), (0, 1), (0, 1)),
        )
    )

    assert result.polynomial.variables == ("x",)
    assert result.polynomial.polynomial.terms == ()
    assert result.degree is None
    assert result.leading_coefficient.as_fraction() == 0
    assert all(item.computed.as_fraction() == 0 for item in result.replay)


def test_node_row_permutation_preserves_polynomial_and_replay_order() -> None:
    jets = (
        _jet(-1, (-23, 2), (25, 2)),
        _jet(0, (-4, 1), (7, 1), (0, 1)),
        _jet(2, (62, 1)),
    )

    forward = hermite_interpolation(_table(*jets))
    reversed_rows = hermite_interpolation(_table(*reversed(jets)))

    assert forward.polynomial == reversed_rows.polynomial
    assert forward.degree == reversed_rows.degree
    assert forward.leading_coefficient == reversed_rows.leading_coefficient
    assert forward.replay == reversed_rows.replay


def test_native_polynomial_value_composes_with_existing_evaluation() -> None:
    result = hermite_interpolation(
        _table(_jet(0, (0, 1), (0, 1)), _jet(1, (1, 1), (2, 1)))
    )
    serialized_polynomial = result.polynomial.model_dump(mode="json")
    request = RationalPolynomialEvaluationRequest.model_validate(
        {"polynomial": serialized_polynomial, "point": {"num": "3", "den": "1"}}
    )

    evaluated = rational_polynomial_evaluate(request)

    assert evaluated.value.as_fraction() == 9


def test_duplicate_nodes_and_incomplete_prefixes_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _table(_jet(1, (1, 1)), _jet(1, (2, 1)))

    with pytest.raises(ValidationError):
        OrdinaryDerivativeJet(
            node=_q(0),
            derivatives=(_derivative(0, 1), _derivative(2, 3)),
        )

    with pytest.raises(ValidationError):
        OrdinaryDerivativeJet(
            node=_q(0),
            derivatives=(_derivative(1, 1),),
        )


def test_total_multiplicity_and_linear_work_boundaries() -> None:
    assert MAX_HERMITE_SYSTEM_CELLS == 32 * 33
    assert MAX_HERMITE_CUBIC_WORK_CELLS == 32**3
    boundary = OrdinaryDerivativeJetTable.model_validate(
        {
            "variable": "x",
            "jets": [
                {
                    "node": {"num": "0", "den": "1"},
                    "derivatives": [
                        {
                            "derivative_order": order,
                            "value": {"num": "0", "den": "1"},
                        }
                        for order in range(32)
                    ],
                }
            ],
        }
    )

    assert sum(len(jet.derivatives) for jet in boundary.jets) == 32
    assert hermite_interpolation(boundary).polynomial.polynomial.terms == ()

    above = boundary.model_dump(mode="json")
    above["jets"][0]["derivatives"].append(
        {"derivative_order": 31, "value": {"num": "0", "den": "1"}}
    )
    with pytest.raises(ValidationError):
        OrdinaryDerivativeJetTable.model_validate(above)


def test_input_rational_digit_boundary_is_documented_and_enforced() -> None:
    at_limit = "9" * 256
    denominator_at_limit = "1" + "0" * 255
    table = OrdinaryDerivativeJetTable.model_validate(
        {
            "variable": "x",
            "jets": [
                {
                    "node": {"num": "1", "den": denominator_at_limit},
                    "derivatives": [
                        {
                            "derivative_order": 0,
                            "value": {"num": at_limit, "den": "1"},
                        }
                    ],
                }
            ],
        }
    )
    assert hermite_interpolation(table).leading_coefficient.num == at_limit

    above_limit = table.model_dump(mode="json")
    above_limit["jets"][0]["derivatives"][0]["value"]["num"] = "9" * 257
    with pytest.raises(ValidationError):
        OrdinaryDerivativeJetTable.model_validate(above_limit)

    above_denominator = table.model_dump(mode="json")
    above_denominator["jets"][0]["node"]["den"] = "1" + "0" * 256
    with pytest.raises(ValidationError):
        OrdinaryDerivativeJetTable.model_validate(above_denominator)

    schema = OrdinaryDerivativeJetTable.model_json_schema()
    jet_properties = schema["$defs"]["OrdinaryDerivativeJet"]["properties"]
    value_properties = schema["$defs"]["OrdinaryDerivativeValue"]["properties"]
    assert (
        f"at most {_MAX_RATIONAL_DIGITS} digits"
        in jet_properties["node"]["description"]
    )
    assert (
        f"at most {_MAX_RATIONAL_DIGITS} digits"
        in value_properties["value"]["description"]
    )


def test_intermediate_growth_boundary_rejects_before_matrix_construction() -> None:
    # At M=32 these adjacent node heights straddle the 32,768-digit
    # fraction-free-minor envelope while every raw rational remains admissible.
    accepted = OrdinaryDerivativeJetTable.model_validate(_full_multiplicity_payload(64))
    assert hermite_interpolation(accepted).polynomial.polynomial.terms == ()
    rejected = OrdinaryDerivativeJetTable.model_validate(_full_multiplicity_payload(65))
    with pytest.raises(OperationDomainValidationError):
        hermite_interpolation(rejected)


def test_aggregate_result_boundary_is_checked_during_request_preflight() -> None:
    # A 256-digit final derivative keeps the minor estimate admissible, while
    # these adjacent node heights straddle the complete 2 MiB result envelope.
    accepted = OrdinaryDerivativeJetTable.model_validate(
        _full_multiplicity_payload(61, last_value_digits=256)
    )
    result = hermite_interpolation(accepted)
    assert (
        len(encode_strict_json(result.model_dump(mode="json")))
        < MAX_HERMITE_RESULT_BYTES
    )
    rejected = OrdinaryDerivativeJetTable.model_validate(
        _full_multiplicity_payload(62, last_value_digits=256)
    )
    with pytest.raises(OperationDomainValidationError):
        hermite_interpolation(rejected)


def test_result_size_is_bounded_and_round_trips() -> None:
    result = hermite_interpolation(
        _table(_jet(0, (0, 1), (0, 1)), _jet(1, (1, 1), (2, 1)))
    )
    serialized = result.model_dump(mode="json")

    assert (
        len(encode_strict_json(result.model_dump(mode="json")))
        < MAX_HERMITE_RESULT_BYTES
    )
    assert HermiteInterpolationResult.model_validate(serialized) == result
