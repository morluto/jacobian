"""Tests for finite abelian subset-sum profiles."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive._finite_abelian_subset_sum import (
    MAX_FINITE_ABELIAN_GROUP_ORDER,
    MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS,
    MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS,
    FiniteAbelianSubsetSumProfileRequest,
    FiniteAbelianSubsetSumProfileResult,
    finite_abelian_subset_sum_profile,
)
from jacobian.math.combinatorics.additive._tools import TOOLS


def _brute_counts(invariant_factors: tuple[int, ...], source: tuple[tuple[int, ...], ...]):
    rank = len(invariant_factors)
    from collections import Counter

    counter: Counter[tuple[int, ...]] = Counter()
    n = len(source)
    for mask in range(1 << n):
        s = tuple(0 for _ in range(rank))
        for i in range(n):
            if mask >> i & 1:
                a = tuple(c % d for c, d in zip(source[i], invariant_factors, strict=True))
                s = tuple((s[j] + a[j]) % invariant_factors[j] for j in range(rank))
        counter[s] += 1
    # ensure all group elements present, sorted
    all_elems = sorted(product(*[range(d) for d in invariant_factors]))
    return [counter.get(elem, 0) for elem in all_elems], all_elems


def test_catalog_contains_finite_abelian_subset_sum():
    assert "additive.subset_sum.finite_abelian_profile.compute" in {
        tool.operation_id for tool in TOOLS
    }


def test_z4_two_elements_full_coverage():
    result = finite_abelian_subset_sum_profile((4,), ((1,), (2,)))
    assert result.support_size == 4
    assert result.covers_group is True
    assert result.total_subsets == "4"
    assert [(e.element, e.multiplicity) for e in result.entries] == [
        ((0,), "1"),
        ((1,), "1"),
        ((2,), "1"),
        ((3,), "1"),
    ]
    # defining invariant: sum multiplicities = 2^k
    total = sum(int(e.multiplicity) for e in result.entries)
    assert total == 1 << len(result.source)
    # zero includes empty subset
    zero_entry = next(e for e in result.entries if e.element == (0,))
    assert int(zero_entry.multiplicity) >= 1


def test_c2_c3_c2xC2_exhaustive_oracle():
    for invariant_factors, source in [
        ((2,), ((1,), (1,))),
        ((3,), ((1,), (1,))),
        ((2, 2), ((1, 0), (0, 1))),
    ]:
        result = finite_abelian_subset_sum_profile(invariant_factors, source)
        brute_counts, all_elems = _brute_counts(invariant_factors, source)
        for entry, expected in zip(result.entries, brute_counts, strict=True):
            assert int(entry.multiplicity) == expected
            assert entry.element in all_elems
        assert result.support_size == sum(1 for c in brute_counts if c > 0)
        assert result.covers_group == (result.support_size == len(all_elems))
        assert int(result.total_subsets) == 1 << len(source)


def test_empty_source_gives_singleton_zero():
    result = finite_abelian_subset_sum_profile((2,), ())
    assert result.support_size == 1
    assert result.covers_group is False
    assert result.total_subsets == "1"
    assert result.entries[0].element == (0,)
    assert result.entries[0].multiplicity == "1"
    assert result.entries[1].multiplicity == "0"


def test_repeated_zero_source_all_zero():
    result = finite_abelian_subset_sum_profile((2,), ((0,), (0,)))
    assert [(e.element, e.multiplicity) for e in result.entries] == [((0,), "4"), ((1,), "0")]
    assert result.support_size == 1
    assert result.covers_group is False


def test_distinguish_equal_group_elements_at_different_indices():
    # Duplicate (1,0) at two positions in C2xC2 -> (1,0) multiplicity 2
    result = finite_abelian_subset_sum_profile((2, 2), ((1, 0), (1, 0)))
    assert [(e.element, e.multiplicity) for e in result.entries] == [
        ((0, 0), "2"),
        ((0, 1), "0"),
        ((1, 0), "2"),
        ((1, 1), "0"),
    ]
    # Verify total 4
    assert sum(int(e.multiplicity) for e in result.entries) == 4


def test_large_coordinates_reduced_modulo():
    # 5 mod 4 =1, 6 mod4=2 same as fixture
    result = finite_abelian_subset_sum_profile((4,), ((5,), (6,)))
    expected = finite_abelian_subset_sum_profile((4,), ((1,), (2,)))
    assert result.entries == expected.entries
    assert result.source == ((1,), (2,))


def test_defining_invariant_dp_transition():
    inv = (2, 2)
    source = ((1, 0), (0, 1), (1, 1))
    result = finite_abelian_subset_sum_profile(inv, source)
    # Each DP step should satisfy c_new(g)=c_old(g)+c_old(g-a_i)
    # Replay stepwise
    from jacobian.math.combinatorics.additive._finite_abelian_subset_sum import (
        _add_elements,
        _all_group_elements,
    )

    elements = _all_group_elements(inv)
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    counts = [0] * len(elements)
    zero = (0, 0)
    counts[elem_to_idx[zero]] = 1
    for a in source:
        a_red = tuple(c % d for c, d in zip(a, inv, strict=True))
        old = counts.copy()
        new = old.copy()
        for g_idx, mult in enumerate(old):
            if mult == 0:
                continue
            target = _add_elements(elements[g_idx], a_red, inv)
            new[elem_to_idx[target]] += mult
        counts = new
    for entry, expected in zip(result.entries, counts, strict=True):
        assert int(entry.multiplicity) == expected
    assert sum(int(e.multiplicity) for e in result.entries) == 1 << len(source)


def test_json_round_trip_and_canonical_source_binding():
    request = FiniteAbelianSubsetSumProfileRequest.model_validate(
        {"invariant_factors": [4], "source": [[1], [2]]}
    )
    assert request.invariant_factors == (4,)
    assert request.source == ((1,), (2,))
    result = finite_abelian_subset_sum_profile(request.invariant_factors, request.source)
    # JSON round-trip via strict parsing
    result_json = result.model_dump_json()
    replay = FiniteAbelianSubsetSumProfileResult.model_validate_json(result_json, strict=True)
    assert replay == result
    # Source is canonicalized (reduced)
    assert replay.source == ((1,), (2,))


def test_group_parent_binding_and_total_multiplicity():
    inv = (6,)
    source = ((2,), (4,), (1,))
    result = finite_abelian_subset_sum_profile(inv, source)
    assert result.invariant_factors == inv
    assert len(result.entries) == 6
    assert int(result.total_subsets) == 8
    assert sum(int(e.multiplicity) for e in result.entries) == 8


def test_rejects_invalid_invariant_factors():
    with pytest.raises(ValidationError, match="divisibility|factor"):
        FiniteAbelianSubsetSumProfileRequest.model_validate(
            {"invariant_factors": [2, 3], "source": [[0, 0]]}
        )
    with pytest.raises(OperationDomainValidationError):
        finite_abelian_subset_sum_profile((2, 3), ((0, 0),))


def test_rejects_rank_mismatch():
    with pytest.raises(ValidationError, match="rank"):
        FiniteAbelianSubsetSumProfileRequest.model_validate(
            {"invariant_factors": [2, 2], "source": [[1]]}
        )
    with pytest.raises(OperationDomainValidationError, match="rank"):
        finite_abelian_subset_sum_profile((4,), ((1, 0),))


def test_rejects_dp_bound():
    # With current bounds, item count 64 is the tighter limit for order 4096 (64*4096=262k <1e6), so DP bound
    # is exercised via validation of source length. Verify that exceeding the item bound is rejected,
    # and that a DP-exhaustive but item-valid request still succeeds.
    with pytest.raises(ValidationError, match="at most 64|too_long"):
        FiniteAbelianSubsetSumProfileRequest.model_validate(
            {"invariant_factors": [64, 64], "source": [[1, 0]] * 65}
        )
    with pytest.raises(OperationDomainValidationError, match="exceeds|item|DP"):
        finite_abelian_subset_sum_profile((64, 64), tuple((1, 0) for _ in range(65)))
    # A maximal item count with small order should succeed
    result = finite_abelian_subset_sum_profile((2,), tuple((1,) for _ in range(64)))
    assert len(result.entries) == 2
    assert int(result.total_subsets) == 1 << 64


def test_covers_group_false_when_not_full():
    result = finite_abelian_subset_sum_profile((4,), ((2,),))
    # subsets: {} ->0, {0} ->2 => elements 0 and2 have 1 each, 1 and3 zero
    assert result.support_size == 2
    assert result.covers_group is False
    assert result.total_subsets == "2"


def test_via_catalog_example_replay():
    from jacobian.catalog.catalog import Catalog

    cat = Catalog.open()
    binding = cat._binding("additive.subset_sum.finite_abelian_profile.compute")
    req = binding.request_type.model_validate(
        {"invariant_factors": [4], "source": [[1], [2]]}
    )
    res = binding.run(req)
    assert res.covers_group is True
    assert res.support_size == 4
