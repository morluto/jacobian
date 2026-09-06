"""Cohomology table identities remain source-bound consumer claims."""

import json

from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.cohomology import group_cohomology, verify_cohomology


def test_forged_cohomology_claims() -> None:
    result = group_cohomology(PermutationGroup(degree=2, generators=((1, 0),)), 2, 1)
    assert verify_cohomology(type(result).model_validate_json(result.model_dump_json()))
    for field, value in (("group_order", 3),):
        payload = result.model_dump()
        payload[field] = value
        assert not verify_cohomology(
            type(result).model_validate_json(json.dumps(payload))
        )
    payload = result.model_dump()
    payload["groups"][0]["betti"] = 0
    assert not verify_cohomology(type(result).model_validate_json(json.dumps(payload)))
    payload = result.model_dump()
    payload["groups"][1]["cochain_dimension"] = 4
    assert not verify_cohomology(type(result).model_validate_json(json.dumps(payload)))
