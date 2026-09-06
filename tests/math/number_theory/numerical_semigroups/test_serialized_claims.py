"""Exact semigroup claims retain sources without proving results on decode."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.numerical_semigroups.operations import (
    element_elasticity_profile,
    global_elasticity,
    summary,
    verify_elasticity,
    verify_element_elasticity,
    verify_summary,
)


@pytest.mark.parametrize("generators", [(1,), (3, 5), (4, 6, 9)])
def test_summary_claim(generators: tuple[int, ...]) -> None:
    result = summary(generators)
    assert type(result.frobenius_number) is int
    assert type(result.conductor) is int
    payload = result.model_dump(mode="json")
    assert isinstance(payload["frobenius_number"], str)
    assert isinstance(payload["conductor"], str)
    assert verify_summary(type(result).model_validate_json(result.model_dump_json()))
    payload = result.model_dump()
    payload["conductor"] = 100
    assert not verify_summary(type(result).model_validate(payload))
    for malformed in ("+1", "01", "x", "-0"):
        payload["frobenius_number"] = malformed
        with pytest.raises(ValidationError):
            type(result).model_validate(payload)


def test_elasticity_claims() -> None:
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier in (
        (global_elasticity((3, 5)), verify_elasticity),
        (element_elasticity_profile((3, 5), 15), verify_element_elasticity),
    ):
        if hasattr(result, "smallest_generator"):
            assert type(result.smallest_generator) is int
        else:
            assert type(result.value) is int
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
        payload = result.model_dump()
        payload["elasticity"] = {"num": 7, "den": 2}
        assert not verifier(type(result).model_validate(payload))
        payload["elasticity"] = {"num": 1, "den": 0}
        with pytest.raises(ValidationError):
            type(result).model_validate(payload)


def test_ratio_alone_does_not_prove_length_extrema() -> None:
    result = element_elasticity_profile((3, 5), 15)
    payload = result.model_dump()
    payload.update(minimum_length=6, maximum_length=10)
    assert not verify_element_elasticity(type(result).model_validate(payload))


def test_verifiers_reject_unadmitted_transported_sources() -> None:
    summary_result = summary((3, 5))
    malformed_source = summary_result.semigroup.model_copy(
        update={"minimal_generators": ("4", "6")}
    )
    assert not verify_summary(
        summary_result.model_copy(update={"semigroup": malformed_source})
    )

    elasticity_result = global_elasticity((3, 5))
    assert not verify_elasticity(
        elasticity_result.model_copy(update={"semigroup": malformed_source})
    )

    element_result = element_elasticity_profile((3, 5), 15)
    assert not verify_element_elasticity(
        element_result.model_copy(update={"value": -1})
    )
