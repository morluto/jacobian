"""Complete bounded sunflower construction for any admitted petal count (#3587)."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    MAX_SUNFLOWER_PETALS,
    SunflowerFamilyRequest,
    construct_sunflower_family,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily


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
    assert result.hypergraph_edges == (("sunflower_0_1_2_3", ("0", "1", "2", "3")),)


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
    with pytest.raises(OperationResourceAdmissionError):
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
    """Petal counts beyond the admitted envelope are refused before expansion."""
    with pytest.raises(OperationResourceAdmissionError):
        construct_sunflower_family(
            SunflowerFamilyRequest(
                source=_family(((0,), (1,))),
                petal_count=MAX_SUNFLOWER_PETALS + 1,
            )
        )
