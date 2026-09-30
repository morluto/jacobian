"""Trusted skeleton production and untrusted decoding have distinct checks."""

from itertools import combinations_with_replacement

import pytest

from jacobian.math.topology.simplicial_sets import from_tables
from jacobian.math.topology.simplicial_sets import skeleton as skeleton_module
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    TruncatedSimplicialMap,
    _require_naturality,
    compose_simplicial_maps,
    identity_simplicial_map,
)
from jacobian.math.topology.simplicial_sets.skeleton import (
    SimplicialSetSkeletonResult,
    simplicial_set_skeleton,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


@pytest.mark.parametrize(
    ("k", "empty"), [(0, False), (1, False), (2, False), (1, True)]
)
def test_native_skeleton_checks_naturality_only_when_decoded(
    k: int, empty: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        from_tables(
            2, ((), (), ()), (((), ()), ((), (), ())), (((),), ((), ()))
        ).simplicial_set
        if empty
        else standard_simplex(2, 2)
    )
    assert source is not None
    checks = 0

    def count_checks(map_value: TruncatedSimplicialMap, *, location: str) -> None:
        nonlocal checks
        checks += 1
        _require_naturality(map_value, location=location)

    monkeypatch.setattr(skeleton_module, "_require_naturality", count_checks)

    result = simplicial_set_skeleton(source, k)

    assert checks == 0
    for degree, level in enumerate(result.skeleton.sets):
        expected = (
            ()
            if empty
            else tuple(
                "(" + ",".join(map(str, simplex)) + ")"
                for simplex in combinations_with_replacement(range(3), degree + 1)
                if len(set(simplex)) <= k + 1
            )
        )
        assert level == expected
    decoded = SimplicialSetSkeletonResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert checks == 1
    assert (
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(
                first=decoded.inclusion, second=identity_simplicial_map(source)
            )
        )
        == decoded.inclusion
    )
