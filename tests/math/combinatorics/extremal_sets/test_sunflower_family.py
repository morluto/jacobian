"""Complete bounded sunflower construction for any admitted petal count (#3587)."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.extremal_sets import _sunflower_r as sunflower_module
from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    MAX_SUNFLOWER_GROUND_SET_SIZE,
    MAX_SUNFLOWER_MEMBERSHIPS,
    SunflowerFamilyRequest,
    SunflowerFamilyResult,
    construct_sunflower_family,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    parameters,
)


def _family(
    members: tuple[tuple[int, ...], ...], ground: int = 8
) -> IndexedFiniteSetFamily:
    return IndexedFiniteSetFamily(ground_set_size=ground, members=members)


def _brute_force(
    members: tuple[tuple[int, ...], ...], petals: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    sets = [set(member) for member in members]
    rows = []
    for indices in itertools.combinations(range(len(sets)), petals):
        cores = {
            frozenset(sets[left] & sets[right])
            for left, right in itertools.combinations(indices, 2)
        }
        if len(cores) == 1:
            rows.append((indices, tuple(sorted(cores.pop()))))
    return rows


def test_issue_fixture_three_petal_sunflowers() -> None:
    """The issue's fixture {01,02,04,05,12,45} with r=3 has four rows, core {0}."""
    source = _family(((0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (4, 5)), ground=6)
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=3)
    )
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        ((0, 1, 2), (0,)),
        ((0, 1, 3), (0,)),
        ((0, 2, 3), (0,)),
        ((1, 2, 3), (0,)),
    ]
    assert result.sunflower_count == 4
    assert result.sunflower_free is False
    assert result.hypergraph.vertices == tuple(str(index) for index in range(6))
    assert result.hypergraph.edges == tuple(
        (row.edge_id, tuple(str(index) for index in row.source_indices))
        for row in result.sunflowers
    )
    assert result.hypergraph_edges == result.hypergraph.edges


def test_four_petals_share_one_core() -> None:
    """Four petals through a common core form exactly one r=4 sunflower."""
    result = construct_sunflower_family(
        SunflowerFamilyRequest(
            source=_family(((0, 1), (0, 2), (0, 3), (0, 4))), petal_count=4
        )
    )
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        ((0, 1, 2, 3), (0,)),
    ]
    assert result.hypergraph_edges == (("sunflower_1", ("0", "1", "2", "3")),)


def test_large_petal_count_keeps_bounded_row_ids() -> None:
    """A 22-petal sunflower keeps its row ID within the hypergraph label limit."""
    members = tuple((0, index + 1) for index in range(22))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(members, ground=23), petal_count=22)
    )
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        (tuple(range(22)), (0,)),
    ]
    assert [row.edge_id for row in result.sunflowers] == ["sunflower_1"]
    assert result.hypergraph_edges == (
        ("sunflower_1", tuple(sorted(str(i) for i in range(22)))),
    )


def test_empty_core_is_a_valid_sunflower() -> None:
    """Disjoint members form a sunflower with the empty core."""
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(((0, 1), (2, 3), (4, 5))), petal_count=3)
    )
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        ((0, 1, 2), ()),
    ]


def test_equal_intersection_cardinalities_with_different_sets_are_not_sunflowers() -> (
    None
):
    """Pairwise sizes agreeing is not the relation; the intersection sets must agree."""
    source = _family(((0, 1), (0, 2), (1, 2)))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=3)
    )
    assert result.sunflowers == ()
    assert result.sunflower_free is True


def test_sunflower_free_family_is_reported_as_such() -> None:
    """A family with no admitted sunflower returns an empty complete family."""
    source = _family(((0, 1), (0, 2), (1, 2), (0, 1, 2)))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=3)
    )
    assert result.sunflowers == ()
    assert result.sunflower_count == 0
    assert result.sunflower_free is True


def test_petal_count_above_the_family_size_is_vacuously_free() -> None:
    """Requesting more petals than members yields no rows, not a failure."""
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(((0,), (1,))), petal_count=3)
    )
    assert result.sunflowers == ()
    assert result.sunflower_free is True


def test_matches_brute_force_on_several_families_and_petal_counts() -> None:
    """Every admitted petal count agrees with exhaustive set-intersection search."""
    families = (
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        ((0,), (1,), (2,), (0, 1)),
    )
    for members in families:
        source = _family(members, ground=4)
        for petals in (2, 3, 4):
            if petals > len(members):
                continue
            result = construct_sunflower_family(
                SunflowerFamilyRequest(source=source, petal_count=petals)
            )
            assert [
                (row.source_indices, row.core) for row in result.sunflowers
            ] == _brute_force(members, petals)


def test_duplicate_source_members_are_rejected_by_the_carrier() -> None:
    """The indexed family carrier already requires pairwise distinct members."""
    with pytest.raises(ValidationError):
        _family(((0, 1), (2, 3), (4, 5), (0, 1)))


def test_petals_sharing_one_point_with_a_larger_member_are_not_a_sunflower() -> None:
    """A petal reaching outside the core breaks the equal-intersection relation."""
    source = _family(((0, 1), (0, 2), (0, 1, 2)))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=3)
    )
    assert result.sunflowers == ()
    assert result.sunflower_free is True


def test_petal_count_below_two_is_rejected() -> None:
    """A single-member subfamily is not a sunflower relation."""
    with pytest.raises(OperationDomainValidationError):
        construct_sunflower_family(
            SunflowerFamilyRequest(source=_family(((0,), (1,))), petal_count=1)
        )


def test_hypergraph_projection_is_empty_but_source_bound_when_no_rows() -> None:
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(((0, 1), (0, 2), (1, 2))), petal_count=3)
    )
    assert result.hypergraph.vertices == ("0", "1", "2")
    assert result.hypergraph.edges == ()


def test_over_bound_petal_count_is_rejected() -> None:
    """Petal counts beyond the shared vertex carrier are refused before expansion."""
    with pytest.raises(OperationResourceAdmissionError):
        construct_sunflower_family(
            SunflowerFamilyRequest(
                source=_family(((0,), (1,))),
                petal_count=MAX_VERTICES + 1,
            )
        )


def test_declared_petals_beyond_the_old_small_slice_remain_exact() -> None:
    """The operation admits any feasible r; r=9 is not a special-case ceiling."""
    source = _family(tuple((0, index) for index in range(1, 10)), ground=10)
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=9)
    )
    assert result.petal_count == 9
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        (tuple(range(9)), (0,)),
    ]


def test_canonical_hypergraph_composes_without_reencoding() -> None:
    result = construct_sunflower_family(
        SunflowerFamilyRequest(
            source=_family(((0, 1), (0, 2), (0, 3)), ground=4), petal_count=3
        )
    )
    profile = parameters(result.hypergraph)
    assert profile.vertex_count == 3
    assert profile.edge_count == 1
    assert profile.uniform_size == 3


def test_schema_advertises_the_complete_declared_petals_envelope() -> None:
    schema = SunflowerFamilyRequest.model_json_schema()
    petal_schema = schema["properties"]["petal_count"]
    assert petal_schema["minimum"] == 2
    assert petal_schema["maximum"] == MAX_VERTICES


def test_projection_uses_canonical_multi_digit_member_order() -> None:
    result = construct_sunflower_family(
        SunflowerFamilyRequest(
            source=_family(tuple((i,) for i in range(11)), ground=11), petal_count=2
        )
    )
    assert result.hypergraph_edges == result.hypergraph.edges
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_output_edge_bound_is_admitted_before_row_construction() -> None:
    """The complete candidate envelope rejects 156 disjoint singleton pairs."""
    source = _family(tuple((index,) for index in range(156)), ground=156)
    with pytest.raises(OperationResourceAdmissionError, match="output bound"):
        construct_sunflower_family(SunflowerFamilyRequest(source=source, petal_count=2))


def test_result_allocation_is_admitted_before_row_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sunflower_module,
        "MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS",
        1,
    )
    source = _family(((0,), (1,)), ground=2)
    with pytest.raises(OperationResourceAdmissionError, match="allocation units"):
        construct_sunflower_family(SunflowerFamilyRequest(source=source, petal_count=2))


def test_exact_candidate_count_at_the_output_boundary_is_admitted() -> None:
    source = _family(tuple((index,) for index in range(155)), ground=155)
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=source, petal_count=2)
    )
    assert result.sunflower_count == 155 * 154 // 2


def test_ground_and_membership_bounds_apply_to_vacuous_requests() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="ground set"):
        construct_sunflower_family(
            SunflowerFamilyRequest(
                source=_family((), ground=MAX_SUNFLOWER_GROUND_SET_SIZE + 1),
                petal_count=2,
            )
        )
    member = tuple(range(MAX_SUNFLOWER_MEMBERSHIPS // 2 + 1))
    second_member = tuple(
        range(MAX_SUNFLOWER_MEMBERSHIPS // 2 - 1, MAX_SUNFLOWER_MEMBERSHIPS)
    )
    with pytest.raises(OperationResourceAdmissionError, match="memberships"):
        construct_sunflower_family(
            SunflowerFamilyRequest(
                source=_family(
                    (member, second_member), ground=MAX_SUNFLOWER_MEMBERSHIPS
                ),
                petal_count=2,
            )
        )


def test_forged_summary_fields_cannot_contradict_rows() -> None:
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(((0, 1), (0, 2), (0, 3))), petal_count=3)
    )
    forged = result.model_dump(mode="json")
    forged["sunflower_count"] = 0
    forged["sunflower_free"] = True
    with pytest.raises(ValidationError):
        SunflowerFamilyResult.model_validate(forged)


def test_forged_core_shape_is_rejected_without_replaying_the_relation() -> None:
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(((0, 1), (0, 2), (0, 3))), petal_count=3)
    )
    forged = result.model_dump(mode="json")
    forged["sunflowers"][0]["core"] = [1, 0]
    with pytest.raises(ValidationError):
        SunflowerFamilyResult.model_validate(forged)


def test_independent_pairwise_intersection_oracle() -> None:
    """The public rows agree with an independently enumerated exact oracle."""
    members = ((0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (4, 5))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(members, ground=6), petal_count=3)
    )
    expected = []
    for indices in itertools.combinations(range(len(members)), 3):
        intersections = [
            set(members[left]).intersection(members[right])
            for left, right in itertools.combinations(indices, 2)
        ]
        if intersections and all(
            intersection == intersections[0] for intersection in intersections[1:]
        ):
            expected.append((indices, tuple(sorted(intersections[0]))))
    assert [(row.source_indices, row.core) for row in result.sunflowers] == expected


def test_wide_single_candidate_checkpoints_inside_pairwise_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        sunflower_module,
        "request_checkpoint",
        lambda label: labels.append(label),
    )
    members = tuple((index,) for index in range(24))
    result = construct_sunflower_family(
        SunflowerFamilyRequest(source=_family(members, ground=24), petal_count=24)
    )
    assert result.sunflower_count == 1
    assert "during sunflower pairwise intersection" in labels

