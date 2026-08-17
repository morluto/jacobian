"""Structural and exact tests for bounded Markov-chain mixing time."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.contracts.markov_chain import MixingTimeRequest, MixingTimeResult
from jacobian.domains.markov_chain.operations import compute_mixing_time
from jacobian.math.markov_chain import mixing_time

type RationalPayload = dict[str, str]
type MatrixPayload = tuple[tuple[RationalPayload, ...], ...]


def _q(numerator: str, denominator: str = "1") -> RationalPayload:
    return {"num": numerator, "den": denominator}


TWO_STATE: MatrixPayload = (
    (_q("1", "2"), _q("1", "2")),
    (_q("1", "4"), _q("3", "4")),
)
UNIFORM_TWO: MatrixPayload = (
    (_q("1", "2"), _q("1", "2")),
    (_q("1", "2"), _q("1", "2")),
)
SWAP: MatrixPayload = (
    (_q("0"), _q("1")),
    (_q("1"), _q("0")),
)
IDENTITY: MatrixPayload = (
    (_q("1"), _q("0")),
    (_q("0"), _q("1")),
)


def _request(
    matrix: MatrixPayload = TWO_STATE,
    *,
    epsilon: tuple[str, str] = ("1", "100"),
    max_steps: int = 64,
) -> MixingTimeRequest:
    return MixingTimeRequest.model_validate(
        {
            "matrix": matrix,
            "epsilon": _q(*epsilon),
            "max_steps": max_steps,
        }
    )


@pytest.mark.parametrize(
    ("epsilon", "mixing_step", "distance"),
    [
        (("1", "100"), 4, Fraction(1, 384)),
        (("1", "6"), 1, Fraction(1, 6)),
        (("1", "1"), 0, Fraction(2, 3)),
    ],
)
def test_two_state_chain_has_exact_known_mixing_time(
    epsilon: tuple[str, str],
    mixing_step: int,
    distance: Fraction,
) -> None:
    result = compute_mixing_time(_request(epsilon=epsilon))
    assert result.status == "FOUND"
    assert result.mixing_time == mixing_step
    assert result.steps_examined == mixing_step
    assert result.max_total_variation_distance is not None
    assert result.max_total_variation_distance.as_fraction() == distance


def test_search_bound_returns_exact_terminal_distance() -> None:
    result = compute_mixing_time(_request(max_steps=3))
    assert result.status == "BOUND_EXCEEDED"
    assert result.mixing_time is None
    assert result.steps_examined == 3
    assert result.max_total_variation_distance is not None
    assert result.max_total_variation_distance.as_fraction() == Fraction(1, 96)
    assert Fraction(1, 96) > Fraction(1, 100)


def test_uniform_chain_mixes_after_one_step() -> None:
    result = compute_mixing_time(_request(UNIFORM_TWO))
    assert result.status == "FOUND"
    assert result.mixing_time == 1
    assert result.max_total_variation_distance is not None
    assert result.max_total_variation_distance.as_fraction() == 0


def test_one_state_chain_is_mixed_at_step_zero() -> None:
    result = compute_mixing_time(_request(((_q("1"),),)))
    assert result.status == "FOUND"
    assert result.mixing_time == 0


@pytest.mark.parametrize("matrix", [SWAP, IDENTITY])
def test_nonergodic_chains_return_typed_outcome(matrix: MatrixPayload) -> None:
    result = compute_mixing_time(_request(matrix, max_steps=4))
    assert result == MixingTimeResult(
        status="NOT_ERGODIC",
        epsilon=_q("1", "100"),
        max_steps=4,
        steps_examined=0,
    )


def test_native_kernel_uses_exact_fraction_values() -> None:
    result = mixing_time(
        (
            (Fraction(1, 2), Fraction(1, 2)),
            (Fraction(1, 4), Fraction(3, 4)),
        ),
        (Fraction(1, 3), Fraction(2, 3)),
        Fraction(1, 100),
        4,
    )
    assert result.mixing_time == 4
    assert result.max_total_variation_distance == Fraction(1, 384)


def test_eight_state_uniform_chain_respects_dimension_boundary() -> None:
    matrix = tuple(tuple(_q("1", "8") for _ in range(8)) for _ in range(8))
    result = compute_mixing_time(_request(matrix))
    assert result.status == "FOUND"
    assert result.mixing_time == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "matrix": ((_q("1", "2"), _q("1", "2")),),
            "epsilon": _q("1", "100"),
        },
        {
            "matrix": (
                (_q("1", "2"), _q("1", "3")),
                (_q("1", "4"), _q("3", "4")),
            ),
            "epsilon": _q("1", "100"),
        },
        {
            "matrix": (
                (_q("-1"), _q("2")),
                (_q("1", "4"), _q("3", "4")),
            ),
            "epsilon": _q("1", "100"),
        },
        {"matrix": TWO_STATE, "epsilon": _q("0")},
        {"matrix": TWO_STATE, "epsilon": _q("2")},
        {"matrix": TWO_STATE, "epsilon": _q("1", "100"), "max_steps": 0},
        {"matrix": TWO_STATE, "epsilon": _q("1", "100"), "max_steps": 257},
        {"matrix": TWO_STATE, "epsilon": _q("1", "100"), "max_steps": True},
    ],
)
def test_request_rejects_invalid_or_unbounded_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MixingTimeRequest.model_validate(payload)


def test_request_rejects_ninth_state() -> None:
    matrix = tuple(tuple(_q("1", "9") for _ in range(9)) for _ in range(9))
    with pytest.raises(ValidationError):
        _request(matrix)


def test_request_rejects_oversized_rational_components() -> None:
    denominator = 10**32
    row = (
        _q("1", str(denominator)),
        _q(str(denominator - 1), str(denominator)),
    )
    with pytest.raises(ValidationError, match="32-digit bound"):
        _request((row, row))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "FOUND",
            "epsilon": _q("1", "100"),
            "max_steps": 4,
            "steps_examined": 4,
            "mixing_time": 4,
            "max_total_variation_distance": _q("1", "10"),
        },
        {
            "status": "BOUND_EXCEEDED",
            "epsilon": _q("1", "100"),
            "max_steps": 4,
            "steps_examined": 4,
            "max_total_variation_distance": _q("1", "1000"),
        },
        {
            "status": "NOT_ERGODIC",
            "epsilon": _q("1", "100"),
            "max_steps": 4,
            "steps_examined": 1,
        },
    ],
)
def test_result_contract_rejects_inconsistent_outcomes(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MixingTimeResult.model_validate(payload)


def test_operation_is_discoverable_with_authoritative_types() -> None:
    from jacobian.domains.markov_chain import markov_chain_operations

    operation = next(
        operation
        for operation in markov_chain_operations()
        if operation.operation_id == "probability.markov_chain.mixing_time.compute"
    )
    assert operation.request_type is MixingTimeRequest
    assert operation.result_type is MixingTimeResult
