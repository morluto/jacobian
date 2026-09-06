"""Multiple-testing claims preserve exact values and hypothesis sources."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.probability.multiple_testing import (
    bh_step_up,
    false_discovery_proportion,
    verify_bh_step_up,
    verify_fdp,
)
from jacobian.math.probability.multiple_testing._models import HypothesisSpec


def test_fdp_source_and_claim() -> None:
    result = false_discovery_proportion(("a", "b", "a"), ("a",))
    assert verify_fdp(type(result).model_validate_json(result.model_dump_json()))
    payload = result.model_dump()
    payload["fdp"] = {"num": "1", "den": "1"}
    assert not verify_fdp(type(result).model_validate(payload))
    payload["fdp"] = {"num": "1", "den": "0"}
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_bh_source_and_claim() -> None:
    result = bh_step_up(
        (
            HypothesisSpec(
                hypothesis_id="a", p_value=CanonicalRational(num="0", den="1")
            ),
        ),
        CanonicalRational(num="1", den="20"),
    )
    assert verify_bh_step_up(type(result).model_validate_json(result.model_dump_json()))
    payload = result.model_dump()
    payload["rejected"] = []
    assert not verify_bh_step_up(type(result).model_validate(payload))
    payload["rejected"] = ["foreign"]
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)
