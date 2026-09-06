"""Action relations are checked explicitly after structural decoding."""

from collections.abc import Callable
from typing import Any

from jacobian.math.groups.actions import (
    burnside_count,
    cycle_index,
    element_cycles,
    polya_inventory,
    verify_burnside_count,
    verify_cycle_index,
    verify_element_cycles,
    verify_polya_inventory,
    verify_subset_family_orbit_profile,
)
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.groups.actions.operations import subset_family_orbit_profile


def test_serialized_action_claims() -> None:
    action = FinitePermutationAction(domain=("a", "b"), generators=((1, 0),))
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier, field, forged in (
        (element_cycles(action, 0), verify_element_cycles, "support", (0,)),
        (cycle_index(action), verify_cycle_index, "group_order", 3),
        (burnside_count(action), verify_burnside_count, "fixed_point_sum", 0),
        (polya_inventory(action, 2), verify_polya_inventory, "terms", (((0, 1), 1),)),
        (
            subset_family_orbit_profile(action, ((0,),)),
            verify_subset_family_orbit_profile,
            "total_full_orbit_size",
            1,
        ),
    ):
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
        payload = result.model_dump()
        payload[field] = forged
        assert not verifier(type(result).model_validate(payload))
