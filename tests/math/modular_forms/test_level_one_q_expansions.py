"""Exact contract tests for the reviewed level-one named q-expansion leaf."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.formal_power_series._operations import (
    compute_power,
    compute_scalar_multiply,
    compute_subtract,
)
from jacobian.math.modular_forms._models import LevelOneNamedQExpansionRequest
from jacobian.math.modular_forms.kernel import MAX_LEVEL_ONE_TRUNCATION_ORDER
from jacobian.math.modular_forms.operations import level_one_named_q_expansion
from jacobian.math.modular_forms.values import LevelOneModularQExpansion


def _integers(expansion: LevelOneModularQExpansion) -> tuple[int, ...]:
    return tuple(
        int(coefficient.as_fraction())
        for coefficient in expansion.q_expansion.coefficients
    )


def test_e4_and_e6_known_normalized_prefixes() -> None:
    assert _integers(level_one_named_q_expansion("E4", 6)) == (
        1,
        240,
        2160,
        6720,
        17520,
        30240,
    )
    assert _integers(level_one_named_q_expansion("E6", 6)) == (
        1,
        -504,
        -16632,
        -122976,
        -532728,
        -1575504,
    )


def test_delta_known_prefix_and_defining_e4_e6_identity() -> None:
    delta = level_one_named_q_expansion("DELTA", 7)
    assert _integers(delta) == (0, 1, -24, 252, -1472, 4830, -6048)
    e4 = level_one_named_q_expansion("E4", 7).q_expansion
    e6 = level_one_named_q_expansion("E6", 7).q_expansion
    derived = compute_scalar_multiply(
        compute_subtract(
            compute_power(e4, 3).result, compute_power(e6, 2).result
        ).result,
        CanonicalRational(num="1", den="1728"),
    ).result
    assert derived == delta.q_expansion


def test_full_public_precision_is_complete_and_carries_parent_metadata() -> None:
    result = level_one_named_q_expansion("DELTA", MAX_LEVEL_ONE_TRUNCATION_ORDER)
    assert result.q_expansion.truncation_order == MAX_LEVEL_ONE_TRUNCATION_ORDER
    assert len(result.q_expansion.coefficients) == MAX_LEVEL_ONE_TRUNCATION_ORDER
    assert result.congruence_subgroup == "SL2Z"
    assert result.level == 1
    assert result.weight == 12
    assert result.space_kind == "CUSP"


def test_order_one_retains_the_known_constant_coefficient() -> None:
    assert _integers(level_one_named_q_expansion("E4", 1)) == (1,)
    assert _integers(level_one_named_q_expansion("DELTA", 1)) == (0,)


def test_request_rejects_precision_above_complete_public_envelope() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 512"):
        LevelOneNamedQExpansionRequest(form="E4", truncation_order=513)


def test_value_rejects_forged_coefficient_during_replay() -> None:
    result = level_one_named_q_expansion("E4", 3)
    payload = result.model_dump()
    q_expansion = payload["q_expansion"]
    q_expansion["coefficients"] = (
        {"num": "1", "den": "1"},
        {"num": "241", "den": "1"},
        {"num": "2160", "den": "1"},
    )
    with pytest.raises(ValidationError, match="does not match"):
        LevelOneModularQExpansion.model_validate(payload)
