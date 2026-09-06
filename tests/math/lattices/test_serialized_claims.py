"""Lattice relation sources and exact scalars survive serialization."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.lattices import (
    IntegerLattice,
    compute_discriminant_group,
    compute_rank_gram,
    compute_saturation,
    compute_sublattice_index,
    verify_discriminant_group,
    verify_rank_gram,
    verify_sublattice_index,
)
from jacobian.math.matrices.values import IntegerMatrix


def test_source_bound_lattice_claims() -> None:
    parent = IntegerLattice(
        ambient_dimension=2, basis=IntegerMatrix(entries=(("1", "0"),))
    )
    child = IntegerLattice(
        ambient_dimension=2, basis=IntegerMatrix(entries=(("2", "0"),))
    )
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier, field, forged in (
        (compute_rank_gram(child), verify_rank_gram, "covolume_rational", False),
        (
            compute_discriminant_group(child),
            verify_discriminant_group,
            "invariant_factors",
            ["8"],
        ),
        (
            compute_sublattice_index(child, parent, IntegerMatrix(entries=(("2",),))),
            verify_sublattice_index,
            "index",
            3,
        ),
    ):
        for key, value in result.model_dump(mode="json").items():
            if key in {"squared_covolume", "invariant_factors"}:
                if key == "invariant_factors":
                    assert all(isinstance(item, str) for item in value)
                else:
                    assert isinstance(value, str)
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
        payload = result.model_dump()
        payload[field] = forged
        assert not verifier(type(result).model_validate(payload))


def test_covolume_is_canonical_and_bound_to_source() -> None:
    lattice = IntegerLattice(
        ambient_dimension=1, basis=IntegerMatrix(entries=(("2",),))
    )
    result = compute_rank_gram(lattice)
    assert result.squared_covolume == 4
    assert result.model_dump(mode="json")["squared_covolume"] == "4"
    payload = result.model_dump()
    payload["lattice"]["basis"]["entries"] = [["3"]]
    assert not verify_rank_gram(type(result).model_validate(payload))
    for malformed in ("anything", "+4", "04", "-0"):
        payload["squared_covolume"] = malformed
        with pytest.raises(ValidationError):
            type(result).model_validate(payload)


def test_integer_claim_scalars_are_native_with_canonical_json() -> None:
    parent = IntegerLattice(ambient_dimension=1, basis=IntegerMatrix(entries=(("1",),)))
    child = IntegerLattice(ambient_dimension=1, basis=IntegerMatrix(entries=(("2",),)))
    claims = (
        (compute_saturation(child), "saturation_index"),
        (
            compute_sublattice_index(child, parent, IntegerMatrix(entries=(("2",),))),
            "index",
        ),
        (compute_discriminant_group(child), "discriminant_order"),
    )

    for claim, field in claims:
        assert type(getattr(claim, field)) is int
        assert isinstance(claim.model_dump(mode="json")[field], str)
        assert type(claim).model_validate_json(claim.model_dump_json()) == claim
        payload = claim.model_dump()
        payload[field] = "01"
        with pytest.raises(ValidationError):
            type(claim).model_validate(payload)
