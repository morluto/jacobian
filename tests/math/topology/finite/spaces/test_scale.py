"""Sparse finite spaces retain their axes beyond the former carrier cap."""

import json
from itertools import product
from typing import Any

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.finite.spaces import operations as native
from jacobian.math.topology.finite.spaces._models import KolmogorovQuotientResult
from jacobian.math.topology.finite.spaces._tools import TOOLS
from jacobian.math.topology.finite.spaces.values import (
    FiniteTopologicalMap,
    FiniteTopologicalSpace,
)


@pytest.mark.parametrize("count", [65, 4_096])
def test_public_discrete_space_retains_large_axis(count: int) -> None:
    space = FiniteTopologicalSpace(
        points=tuple(f"p{i}" for i in range(count)),
        preorder=tuple((i,) for i in range(count)),
    )
    subset = tuple(range(0, count, 2))
    for name in ("interior", "closure", "boundary"):
        tool = next(
            t for t in TOOLS if t.operation_id == f"topology.finite.{name}.compute"
        )
        request = tool.request_type.model_validate_json(
            json.dumps({"space": space.model_dump(mode="json"), "subset": subset})
        )
        result = tool.run(request)
        claim = tool.result_type.model_validate_json(result.model_dump_json())
        assert claim.space == space
        assert getattr(claim, name).indices == (() if name == "boundary" else subset)
        assert getattr(native, f"verify_{name}")(claim)
    assert native.minimal_neighbourhoods(space) == space.preorder
    quotient = native.kolmogorov_quotient(space)
    claim = KolmogorovQuotientResult.model_validate_json(quotient.model_dump_json())
    assert claim.quotient_map.target == space
    assert native.verify_kolmogorov_quotient(claim)
    assert native.continuous_check(claim.quotient_map)


def test_chain_uses_incidence_work_and_exact_open_closed_sets() -> None:
    count = 256
    space = native.from_preorder(
        tuple(map(str, range(count))),
        tuple(tuple(range(i + 1)) for i in range(count)),
    )
    upper = frozenset(range(128, count))
    assert native.interior(space, upper) == upper
    assert native.closure(space, upper) == frozenset(range(count))
    assert native.boundary(space, upper) == frozenset(range(128))
    assert native.minimal_neighbourhoods(space) == tuple(
        tuple(range(i, count)) for i in range(count)
    )


def test_dense_space_is_refused_before_preorder_expansion() -> None:
    count = 512
    space = FiniteTopologicalSpace(
        points=tuple(map(str, range(count))),
        preorder=(tuple(range(count)),) * count,
    )
    tool = next(t for t in TOOLS if t.operation_id == "topology.finite.closure.compute")
    request = tool.request_type.model_validate_json(
        json.dumps({"space": space.model_dump(mode="json"), "subset": [0]})
    )
    with pytest.raises(OperationResourceAdmissionError, match="bitset work"):
        tool.run(request)


def test_sparse_equivalence_quotient_at_bit_work_boundary() -> None:
    count = 4_096
    space = native.from_preorder(
        tuple(map(str, range(count))),
        tuple((2 * (i // 2), 2 * (i // 2) + 1) for i in range(count)),
    )
    result = native.kolmogorov_quotient(space)
    assert result.quotient_map.point_map == tuple(i // 2 for i in range(count))
    target = result.quotient_map.target
    assert target.preorder == tuple((i,) for i in range(count // 2))
    assert native.verify_kolmogorov_quotient(
        KolmogorovQuotientResult.model_validate_json(result.model_dump_json())
    )


def test_quotient_verifier_accepts_a_reordered_target_axis() -> None:
    source = native.from_preorder(("a", "b", "c"), ((0,), (1,), (2,)))
    target = native.from_preorder(("c", "b", "a"), ((0,), (1,), (2,)))
    claim = KolmogorovQuotientResult(
        quotient_map=FiniteTopologicalMap(
            source=source, target=target, point_map=(2, 1, 0)
        )
    )
    assert native.verify_kolmogorov_quotient(claim)
    forged = claim.model_copy(
        update={
            "quotient_map": claim.quotient_map.model_copy(
                update={"point_map": (1, 2, 0)}
            )
        }
    )
    assert not native.verify_kolmogorov_quotient(forged)


def test_bitset_admission_and_interior_match_all_three_point_preorders() -> None:
    count = 3
    off_diagonal = [(i, j) for i in range(count) for j in range(count) if i != j]
    for bits in product((False, True), repeat=len(off_diagonal)):
        relation = {(i, i) for i in range(count)} | {
            pair for pair, present in zip(off_diagonal, bits, strict=True) if present
        }
        if any(
            (a, c) not in relation for a, b in relation for b2, c in relation if b == b2
        ):
            continue
        rows = tuple(
            tuple(j for j in range(count) if (j, i) in relation) for i in range(count)
        )
        space = native.from_preorder(("a", "b", "c"), rows)
        for membership in product((False, True), repeat=count):
            subset = frozenset(i for i, present in enumerate(membership) if present)
            expected = frozenset(
                i
                for i in range(count)
                if all(j in subset for a, j in relation if a == i)
            )
            assert native.interior(space, subset) == expected


@pytest.mark.parametrize(
    "name", ["interior", "closure", "boundary", "continuity", "kolmogorov_quotient"]
)
def test_true_serialized_claim_preserves_resource_refusal(name: str) -> None:
    payload: dict[str, Any]
    count = 512
    space = FiniteTopologicalSpace(
        points=tuple(map(str, range(count))),
        preorder=(tuple(range(count)),) * count,
    )
    if name in ("interior", "closure", "boundary"):
        tool = next(
            t for t in TOOLS if t.operation_id == f"topology.finite.{name}.compute"
        )
        subset = {"space": space.model_dump(mode="json"), "indices": []}
        payload = {
            "space": space.model_dump(mode="json"),
            "subset": subset,
            name: subset,
        }
    elif name == "continuity":
        tool = next(
            t
            for t in TOOLS
            if t.operation_id == "topology.finite.continuity_check.compute"
        )
        payload = {
            "point_map": {
                "source": space.model_dump(mode="json"),
                "target": space.model_dump(mode="json"),
                "point_map": list(range(count)),
            },
            "is_continuous": True,
        }
    else:
        tool = next(
            t
            for t in TOOLS
            if t.operation_id == "topology.finite.kolmogorov_quotient.compute"
        )
        payload = {
            "quotient_map": {
                "source": space.model_dump(mode="json"),
                "target": {"points": ["0"], "preorder": [[0]]},
                "point_map": [0] * count,
            }
        }
    claim = tool.result_type.model_validate_json(json.dumps(payload))
    with pytest.raises(OperationResourceAdmissionError):
        getattr(native, f"verify_{name}")(claim)
