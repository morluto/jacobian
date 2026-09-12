"""Behavioral tests for the pinned finite Berry--Esseen operation."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability._berry_esseen import (
    BERRY_ESSEEN_CONSTANT,
    BERRY_ESSEEN_THEOREM_VARIANT,
    BerryEsseenRequest,
    BerryEsseenResult,
    berry_esseen_bound,
)
from jacobian.math.probability._berry_esseen_tools import (
    BERRY_ESSEEN_OPERATION,
)


def _distribution(*atoms: tuple[int, Fraction]) -> dict[str, object]:
    return {
        "atoms": [
            {
                "value": {"num": value, "den": 1},
                "probability": {
                    "num": probability.numerator,
                    "den": probability.denominator,
                },
            }
            for value, probability in atoms
        ]
    }


def _request(*summands: dict[str, object]) -> BerryEsseenRequest:
    return BerryEsseenRequest.model_validate({"summands": summands})


def test_fair_bernoulli_returns_exact_moments_and_exact_bound() -> None:
    result = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))))
    )

    assert result.theorem_variant == BERRY_ESSEEN_THEOREM_VARIANT
    assert result.universal_constant.as_fraction() == BERRY_ESSEEN_CONSTANT
    assert result.total_mean.as_fraction() == Fraction(1, 2)
    assert result.total_variance.as_fraction() == Fraction(1, 4)
    assert result.total_third_absolute_central_moment.as_fraction() == Fraction(1, 8)
    assert result.bound_squared.as_fraction() == Fraction(196, 625)
    assert result.bound_lower.as_fraction() == Fraction(14, 25)
    assert result.bound_upper.as_fraction() == Fraction(14, 25)


def test_asymmetric_independent_summands_use_exact_non_iid_aggregate() -> None:
    result = berry_esseen_bound(
        _request(
            _distribution((0, Fraction(1, 3)), (2, Fraction(2, 3))),
            _distribution((-1, Fraction(1, 4)), (3, Fraction(3, 4))),
        )
    )

    assert result.total_mean.as_fraction() == Fraction(10, 3)
    assert result.total_variance.as_fraction() == Fraction(35, 9)
    assert result.total_third_absolute_central_moment.as_fraction() == Fraction(
        80, 81
    ) + Fraction(15, 2)
    assert result.bound_lower.as_fraction() <= result.bound_upper.as_fraction()
    assert (
        result.bound_lower.as_fraction() ** 2
        <= result.bound_squared.as_fraction()
        <= result.bound_upper.as_fraction() ** 2
    )


def test_iid_repetition_has_the_expected_inverse_square_root_scaling() -> None:
    fair_bit = _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2)))
    one = berry_esseen_bound(_request(fair_bit))
    four = berry_esseen_bound(_request(fair_bit, fair_bit, fair_bit, fair_bit))

    assert four.total_mean.as_fraction() == 4 * one.total_mean.as_fraction()
    assert four.total_variance.as_fraction() == 4 * one.total_variance.as_fraction()
    assert four.bound_squared.as_fraction() == one.bound_squared.as_fraction() / 4


def test_serialized_result_preserves_source_and_bound() -> None:
    result = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 3)), (2, Fraction(2, 3))))
    )
    restored = BerryEsseenResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.source == result.source


def test_zero_variance_summand_is_rejected_at_operation_boundary() -> None:
    request = _request(_distribution((7, Fraction(1, 1))))

    with pytest.raises(OperationDomainValidationError, match="positive variance"):
        berry_esseen_bound(request)


def test_probability_normalization_is_rejected_at_operation_boundary() -> None:
    request = _request(_distribution((0, Fraction(1, 1)), (1, Fraction(1, 1))))

    with pytest.raises(OperationDomainValidationError, match="sum exactly to 1"):
        berry_esseen_bound(request)


def test_total_atom_admission_precedes_moment_expansion() -> None:
    atom_count = 16_385
    request = _request(
        _distribution(
            *((index, Fraction(1, atom_count)) for index in range(atom_count))
        )
    )

    with pytest.raises(OperationResourceAdmissionError, match="input atoms"):
        berry_esseen_bound(request)


def test_operation_declaration_pins_constant_and_variant() -> None:
    assert BERRY_ESSEEN_OPERATION.operation_id.endswith("berry_esseen_05600.compute")
    assert "0.5600" in BERRY_ESSEEN_OPERATION.description
    assert BERRY_ESSEEN_OPERATION.request_type is BerryEsseenRequest
    assert BERRY_ESSEEN_OPERATION.result_type is BerryEsseenResult
