"""Dispatch execution tests for finite group-action orbit profiles."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation

_ACTION = {"domain": ["a", "b", "c"], "generators": [[1, 2, 0]]}


def test_math_run_executes_subset_family_orbit_profile() -> None:
    result = invoke_operation(
        "group_action.subset_family.orbit_profile.compute",
        {"action": _ACTION, "subsets": [[1], [0], [2]]},
        Catalog.open(),
    )

    assert result.output["family_size"] == 3
    assert result.output["group_order"] == 3
    assert result.output["is_union_of_complete_orbits"] is True
    assert len(result.output["rows"]) == 1
    assert result.output["rows"][0]["source_indices"] == [0, 1, 2]
    assert result.output["rows"][0]["orbit_size"] == 3
    assert result.output["rows"][0]["stabilizer_size"] == 1
