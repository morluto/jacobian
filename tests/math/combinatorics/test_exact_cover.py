"""Contract tests for bounded generalized exact cover."""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.exact_cover import (
    MAX_EXACT_COVER_INCIDENCES,
    MAX_EXACT_COVER_PRIMARY_ITEMS,
    MAX_EXACT_COVER_ROWS,
    MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
    ExactCoverRow,
    GeneralizedExactCoverInstance,
    GeneralizedExactCoverRequest,
    GeneralizedExactCoverResult,
    find_generalized_exact_cover,
)


def _instance(
    *,
    primary: tuple[str, ...],
    secondary: tuple[str, ...] = (),
    rows: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> GeneralizedExactCoverInstance:
    return GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=secondary,
        rows=tuple(ExactCoverRow(row_id=row_id, items=items) for row_id, items in rows),
    )


def _solve(
    instance: GeneralizedExactCoverInstance,
    *,
    search_node_limit: int = 100_000,
) -> GeneralizedExactCoverResult:
    return find_generalized_exact_cover(
        GeneralizedExactCoverRequest(
            instance=instance,
            search_node_limit=search_node_limit,
        )
    )


def _multiplicities(result: GeneralizedExactCoverResult) -> dict[str, int]:
    assert result.item_multiplicities is not None
    return {item.item_id: item.multiplicity for item in result.item_multiplicities}


def test_knuth_exact_cover_known_answer_is_deterministic() -> None:
    # The six-row example from Knuth's Algorithm X has the unique cover B,D,F.
    instance = _instance(
        primary=("1", "2", "3", "4", "5", "6", "7"),
        rows=(
            ("A", ("1", "4", "7")),
            ("B", ("1", "4")),
            ("C", ("4", "5", "7")),
            ("D", ("3", "5", "6")),
            ("E", ("2", "3", "6", "7")),
            ("F", ("2", "7")),
        ),
    )

    first = _solve(instance)
    second = _solve(instance)

    assert first.status == "FOUND"
    assert first.selected_row_ids == ("B", "D", "F")
    assert _multiplicities(first) == dict.fromkeys(instance.primary_items, 1)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_secondary_items_are_optional_but_never_reusable() -> None:
    found = _solve(
        _instance(
            primary=("p", "q"),
            secondary=("s", "t"),
            rows=(
                ("r1", ("p", "s")),
                ("r2", ("q", "t")),
            ),
        )
    )
    assert found.status == "FOUND"
    assert _multiplicities(found) == {"p": 1, "q": 1, "s": 1, "t": 1}

    optional = _solve(
        _instance(
            primary=("p",),
            secondary=("s",),
            rows=(("r", ("p",)),),
        )
    )
    assert optional.status == "FOUND"
    assert _multiplicities(optional) == {"p": 1, "s": 0}

    conflict = _solve(
        _instance(
            primary=("p", "q"),
            secondary=("s",),
            rows=(
                ("r1", ("p", "s")),
                ("r2", ("q", "s")),
            ),
        )
    )
    assert conflict.status == "NO_COVER"
    assert conflict.selected_row_ids is None
    assert conflict.item_multiplicities is None


def test_node_limit_returns_unknown_not_no_cover() -> None:
    instance = _instance(
        primary=("p", "q"),
        secondary=("s",),
        rows=(
            ("r1", ("p", "s")),
            ("r2", ("q", "s")),
        ),
    )

    limited = _solve(instance, search_node_limit=1)
    complete = _solve(instance, search_node_limit=2)

    assert limited.status == "UNKNOWN"
    assert limited.selected_row_ids is None
    assert complete.status == "NO_COVER"
    # UNKNOWN is an operational outcome bound to this deterministic limit.
    assert type(limited).model_validate(limited.model_dump()) == limited


def test_empty_primary_domain_has_the_empty_cover() -> None:
    instance = _instance(
        primary=(),
        secondary=("resource",),
        rows=(("irrelevant", ("resource",)),),
    )

    result = _solve(instance, search_node_limit=1)

    assert result.status == "FOUND"
    assert result.selected_row_ids == ()
    assert _multiplicities(result) == {"resource": 0}


def test_uncovered_primary_item_proves_no_cover_at_the_root() -> None:
    result = _solve(_instance(primary=("required",)), search_node_limit=1)

    assert result.status == "NO_COVER"


def _brute_force_feasible(instance: GeneralizedExactCoverInstance) -> bool:
    items = (*instance.primary_items, *instance.secondary_items)
    for chosen_count in range(len(instance.rows) + 1):
        for chosen in combinations(instance.rows, chosen_count):
            counts = dict.fromkeys(items, 0)
            for row in chosen:
                for item in row.items:
                    counts[item] += 1
            if all(counts[item] == 1 for item in instance.primary_items) and all(
                counts[item] <= 1 for item in instance.secondary_items
            ):
                return True
    return False


def test_small_incidence_families_match_independent_exhaustive_oracle() -> None:
    item_subsets = tuple(
        tuple(item for bit, item in enumerate(("p", "q", "s")) if mask >> bit & 1)
        for mask in range(1, 8)
    )
    all_rows = tuple(
        (f"r{index}", items) for index, items in enumerate(item_subsets, start=1)
    )

    # Exhaust every family of the seven possible nonempty rows on two primary
    # items and one secondary item. This oracle does not share Algorithm X.
    for family_mask in range(1 << len(all_rows)):
        instance = _instance(
            primary=("p", "q"),
            secondary=("s",),
            rows=tuple(
                row for index, row in enumerate(all_rows) if family_mask >> index & 1
            ),
        )
        result = _solve(instance, search_node_limit=1_000)
        expected_status = "FOUND" if _brute_force_feasible(instance) else "NO_COVER"
        assert result.status == expected_status


def test_item_and_row_relabelling_preserves_feasibility() -> None:
    original = _instance(
        primary=("p", "q"),
        secondary=("s",),
        rows=(
            ("r1", ("p", "s")),
            ("r2", ("q",)),
        ),
    )
    relabelled = _instance(
        primary=("alpha", "omega"),
        secondary=("middle",),
        rows=(
            ("left", ("alpha", "middle")),
            ("right", ("omega",)),
        ),
    )

    original_result = _solve(original)
    relabelled_result = _solve(relabelled)

    assert original_result.status == relabelled_result.status == "FOUND"
    assert sorted(_multiplicities(original_result).values()) == sorted(
        _multiplicities(relabelled_result).values()
    )


def test_erdos_743_k3_packing_fixture_and_planted_overfull_mutation() -> None:
    """Replay the compact K3 controls from the pinned Erdős 743 certificate.

    Source: techno-optimist/erdos-frontier-atlas, commit
    0394e3d3b249439ffabec7d96a3311aa441651b8,
    certificates/erdos-743/pack.c selftest. The positive instance selects one
    embedding of T2 and T3 with edge-disjoint supports; the planted mutation
    asks for two three-vertex tree supports, whose four edge incidences cannot
    fit in the three edges of K3.
    """

    packing = _instance(
        primary=("tree:2", "tree:3"),
        secondary=("edge:01", "edge:02", "edge:12"),
        rows=(
            ("t2:01", ("edge:01", "tree:2")),
            ("t2:02", ("edge:02", "tree:2")),
            ("t2:12", ("edge:12", "tree:2")),
            ("t3:center0", ("edge:01", "edge:02", "tree:3")),
            ("t3:center1", ("edge:01", "edge:12", "tree:3")),
            ("t3:center2", ("edge:02", "edge:12", "tree:3")),
        ),
    )
    packed = _solve(packing)
    assert packed.status == "FOUND"
    assert _multiplicities(packed) == dict.fromkeys(
        (*packing.primary_items, *packing.secondary_items), 1
    )

    overfull = _instance(
        primary=("tree:3:a", "tree:3:b"),
        secondary=("edge:01", "edge:02", "edge:12"),
        rows=tuple(
            (f"{tree}:{center}", (*edges, f"tree:3:{tree}"))
            for tree in ("a", "b")
            for center, edges in (
                ("center0", ("edge:01", "edge:02")),
                ("center1", ("edge:01", "edge:12")),
                ("center2", ("edge:02", "edge:12")),
            )
        ),
    )
    assert _solve(overfull).status == "NO_COVER"


def test_result_validation_reconstructs_witness_and_replays_exact_negative() -> None:
    cover = _instance(
        primary=("p", "q"),
        secondary=("s",),
        rows=(
            ("r1", ("p", "s")),
            ("r2", ("q",)),
        ),
    )
    found = _solve(cover)
    payload = found.model_dump(mode="json")
    payload["item_multiplicities"][-1]["multiplicity"] = 0
    with pytest.raises(ValidationError, match="reconstruct"):
        GeneralizedExactCoverResult.model_validate(payload)

    mutated_source = found.model_dump(mode="json")
    mutated_source["instance"]["rows"][0]["items"] = ["p"]
    with pytest.raises(ValidationError, match="reconstruct"):
        GeneralizedExactCoverResult.model_validate(mutated_source)

    with pytest.raises(ValidationError, match="deterministic replay"):
        GeneralizedExactCoverResult(
            instance=cover,
            search_node_limit=100,
            status="NO_COVER",
        )

    no_cover = _instance(
        primary=("p", "q"),
        secondary=("s",),
        rows=(
            ("r1", ("p", "s")),
            ("r2", ("q", "s")),
        ),
    )
    # UNKNOWN is a durable non-conclusion, not a claim that validation must
    # reproduce with one private branching implementation.
    unknown = GeneralizedExactCoverResult(
        instance=no_cover,
        search_node_limit=1,
        status="UNKNOWN",
    )
    assert type(unknown).model_validate(unknown.model_dump()) == unknown


def test_contract_rejects_duplicate_undeclared_and_noncanonical_incidences() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        ExactCoverRow(row_id="r", items=("p", "p"))
    with pytest.raises(ValidationError, match="sorted and unique"):
        ExactCoverRow(row_id="r", items=("q", "p"))
    with pytest.raises(ValidationError, match="must be declared"):
        _instance(primary=("p",), rows=(("r", ("q",)),))
    with pytest.raises(ValidationError, match="must be disjoint"):
        _instance(primary=("p",), secondary=("p",))
    with pytest.raises(ValidationError, match="sorted and unique"):
        _instance(
            primary=("p",),
            rows=(
                ("same", ("p",)),
                ("same", ("p",)),
            ),
        )
    with pytest.raises(ValidationError, match="row IDs must be sorted"):
        _instance(
            primary=("p",),
            rows=(
                ("z", ("p",)),
                ("a", ("p",)),
            ),
        )

    # Equal item sets are distinct candidate rows when their explicit IDs differ.
    repeated_item_set = _solve(
        _instance(
            primary=("p",),
            rows=(
                ("first", ("p",)),
                ("second", ("p",)),
            ),
        )
    )
    assert repeated_item_set.selected_row_ids == ("first",)


def test_row_and_incidence_boundaries_are_admitted_before_search() -> None:
    maximum_rows = tuple(
        ExactCoverRow(row_id=f"r{index:04d}", items=("p",))
        for index in range(MAX_EXACT_COVER_ROWS)
    )
    row_boundary = GeneralizedExactCoverInstance(
        primary_items=("p",), secondary_items=(), rows=maximum_rows
    )
    assert _solve(row_boundary, search_node_limit=2).status == "FOUND"
    with pytest.raises(ValidationError, match="at most 4096 items"):
        GeneralizedExactCoverInstance(
            primary_items=("p",),
            secondary_items=(),
            rows=(
                *maximum_rows,
                ExactCoverRow(row_id="r4096", items=("p",)),
            ),
        )

    primary = tuple(f"p{index:03d}" for index in range(128))
    secondary = tuple(f"s{index:03d}" for index in range(128))
    all_items = (*primary, *secondary)
    incidence_rows = tuple(
        ExactCoverRow(row_id=f"r{index:03d}", items=all_items)
        for index in range(MAX_EXACT_COVER_INCIDENCES // len(all_items))
    )
    incidence_boundary = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=secondary,
        rows=incidence_rows,
    )
    assert sum(len(row.items) for row in incidence_boundary.rows) == (
        MAX_EXACT_COVER_INCIDENCES
    )
    assert _solve(incidence_boundary, search_node_limit=2).status == "FOUND"
    with pytest.raises(ValidationError, match="incidence count"):
        GeneralizedExactCoverInstance(
            primary_items=primary,
            secondary_items=secondary,
            rows=(
                *incidence_rows,
                ExactCoverRow(row_id="r256", items=("p000",)),
            ),
        )


def test_combined_item_bound_is_not_hidden_by_per_field_limits() -> None:
    primary = tuple(f"p{index:03d}" for index in range(MAX_EXACT_COVER_PRIMARY_ITEMS))
    secondary = tuple(f"s{index:03d}" for index in range(129))

    with pytest.raises(ValidationError, match="at most 256 items"):
        GeneralizedExactCoverInstance(
            primary_items=primary,
            secondary_items=secondary,
            rows=(),
        )


def test_129_primary_one_row_instance_uses_the_aggregate_item_envelope() -> None:
    primary = tuple(f"p{index:03d}" for index in range(129))
    instance = _instance(primary=primary, rows=(("all", primary),))

    result = _solve(instance, search_node_limit=2)

    assert result.status == "FOUND"
    assert result.selected_row_ids == ("all",)
    assert _multiplicities(result) == dict.fromkeys(primary, 1)


def test_selected_row_output_reaches_the_256_primary_boundary() -> None:
    primary = tuple(f"p{index:03d}" for index in range(256))
    rows = tuple((f"r{index:03d}", (item,)) for index, item in enumerate(primary))

    result = _solve(_instance(primary=primary, rows=rows), search_node_limit=257)

    assert result.status == "FOUND"
    assert result.selected_row_ids == tuple(row_id for row_id, _items in rows)
    assert len(result.item_multiplicities or ()) == 256


def test_search_and_retained_output_boundaries_are_preflighted() -> None:
    small = _instance(primary=("p",), rows=(("r", ("p",)),))
    assert GeneralizedExactCoverRequest(
        instance=small,
        search_node_limit=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
    )
    with pytest.raises(ValidationError, match="less than or equal to 100000"):
        GeneralizedExactCoverRequest(
            instance=small,
            search_node_limit=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS + 1,
        )

    # The item/row/incidence counts alone do not bound UTF-8 source bytes.
    # This canonical instance fits every combinatorial count but its retained
    # source would exceed the transport's identical output limit.
    prefix = "🟦" * 60
    primary = tuple(f"{prefix}p{index:03d}" for index in range(128))
    secondary = tuple(f"{prefix}s{index:03d}" for index in range(128))
    all_items = (*primary, *secondary)
    large_source = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=secondary,
        rows=tuple(
            ExactCoverRow(row_id=f"r{index:03d}", items=all_items)
            for index in range(MAX_EXACT_COVER_INCIDENCES // len(all_items))
        ),
    )
    with pytest.raises(ValidationError, match="canonical output limit"):
        GeneralizedExactCoverRequest(instance=large_source)


def test_schema_publishes_the_exact_cover_contract() -> None:
    schema = GeneralizedExactCoverRequest.model_json_schema()
    instance_schema = schema["$defs"]["GeneralizedExactCoverInstance"]
    result_schema = GeneralizedExactCoverResult.model_json_schema()
    assert instance_schema["properties"]["primary_items"]["maxItems"] == 256
    assert instance_schema["properties"]["rows"]["maxItems"] == 4_096
    assert schema["properties"]["search_node_limit"]["maximum"] == 100_000
    assert (
        result_schema["properties"]["selected_row_ids"]["anyOf"][0]["maxItems"] == 256
    )
    assert "NO_COVER" in schema["description"]
    assert "UNKNOWN" in schema["description"]
