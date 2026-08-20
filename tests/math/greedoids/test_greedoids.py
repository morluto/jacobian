"""Tests for greedoid operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math import greedoids
from jacobian.math.greedoids import FiniteFeasibleSetSystem
from jacobian.math.greedoids._models import (
    BasesRequest,
    BasicWordProfileRequest,
    ConvexGeometryRequest,
    RankRequest,
    RecognizeRequest,
)
from jacobian.math.greedoids._operations import (
    compute_bases,
    compute_basic_word_profile,
    compute_convex_geometry,
    compute_rank,
    compute_recognize,
)
from jacobian.math.greedoids._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_element_antimatroid() -> FiniteFeasibleSetSystem:
    """A full-support antimatroid on E={a,b}: F = {empty, {a}, {b}, {a,b}}."""
    return FiniteFeasibleSetSystem(
        ground=("a", "b"),
        feasible=((), (0,), (1,), (0, 1)),
    )


def _path_greedoid() -> FiniteFeasibleSetSystem:
    """A path greedoid on 3 vertices a-b-c with edge ab, bc.

    Ground = {ab, bc}; feasible = empty, {ab}, {bc}, {ab,bc}.
    This is a valid greedoid (accessible, exchange) that is NOT an antimatroid:
    {ab} union {bc} = {ab,bc} is feasible, so union-closed holds; here it is also
    union-closed. Use a branched greedoid that is not union-closed below.
    """
    return FiniteFeasibleSetSystem(
        ground=("ab", "bc"),
        feasible=((), (0,), (1,), (0, 1)),
    )


def _non_greedoid_missing_empty() -> FiniteFeasibleSetSystem:
    return FiniteFeasibleSetSystem(
        ground=("a", "b"),
        feasible=((0,), (1,), (0, 1)),
    )


def _non_greedoid_inaccessible() -> FiniteFeasibleSetSystem:
    """{a,b} is feasible but neither {a} nor {b} is, violating accessibility."""
    return FiniteFeasibleSetSystem(
        ground=("a", "b"),
        feasible=((), (0, 1)),
    )


def _non_greedoid_exchange() -> FiniteFeasibleSetSystem:
    """Accessibility holds but exchange fails: {a,b} > {c} with no augmenting element."""
    return FiniteFeasibleSetSystem(
        ground=("a", "b", "c"),
        feasible=((), (0,), (1,), (0, 1), (2,)),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "greedoid.recognize.compute",
        "greedoid.rank.compute",
        "greedoid.bases.compute",
        "greedoid.basic_word.profile.compute",
        "greedoid.convex_geometry.compute",
    }


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


class TestRecognize:
    def test_two_element_antimatroid_is_greedoid(self) -> None:
        result = compute_recognize(RecognizeRequest(system=_two_element_antimatroid()))
        assert result.status == "GREEDOID"
        assert result.rank == 2
        assert result.bases == ((0, 1),)

    def test_path_greedoid_is_greedoid(self) -> None:
        result = compute_recognize(RecognizeRequest(system=_path_greedoid()))
        assert result.status == "GREEDOID"
        assert result.rank == 2

    def test_missing_empty_set(self) -> None:
        result = compute_recognize(
            RecognizeRequest(system=_non_greedoid_missing_empty())
        )
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "missing_empty_set"

    def test_inaccessible_feasible_set(self) -> None:
        result = compute_recognize(
            RecognizeRequest(system=_non_greedoid_inaccessible())
        )
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "inaccessible_feasible_set"
        assert result.feasible_set == (0, 1)

    def test_exchange_violation(self) -> None:
        result = compute_recognize(RecognizeRequest(system=_non_greedoid_exchange()))
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "exchange_violation"


# ---------------------------------------------------------------------------
# Rank and bases
# ---------------------------------------------------------------------------


class TestRankAndBases:
    def test_rank_of_full_ground(self) -> None:
        result = compute_rank(RankRequest(system=_two_element_antimatroid()))
        assert result.rank == 2

    def test_rank_of_subset(self) -> None:
        # r({a}) = 1 because {a} is feasible and {a} is the largest feasible
        # subset of {a}.
        result = compute_rank(
            RankRequest(system=_two_element_antimatroid(), subset=(0,))
        )
        assert result.rank == 1

    def test_rank_of_empty_subset(self) -> None:
        result = compute_rank(RankRequest(system=_two_element_antimatroid(), subset=()))
        assert result.rank == 0

    def test_bases_of_full_ground(self) -> None:
        result = compute_bases(BasesRequest(system=_two_element_antimatroid()))
        assert result.rank == 2
        assert result.bases == ((0, 1),)

    def test_bases_of_subset(self) -> None:
        # Bases of {a} = {{a}} (rank 1).
        result = compute_bases(
            BasesRequest(system=_two_element_antimatroid(), subset=(0,))
        )
        assert result.rank == 1
        assert result.bases == ((0,),)


# ---------------------------------------------------------------------------
# Basic word profile
# ---------------------------------------------------------------------------


class TestBasicWordProfile:
    def test_full_basic_word(self) -> None:
        result = compute_basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0, 1))
        )
        assert result.status == "BASIC_WORD"
        assert result.is_full is True
        assert result.rank == 2

    def test_prefix_basic_word(self) -> None:
        # Word (0,) is a basic word of length 1; not full.
        result = compute_basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0,))
        )
        assert result.status == "BASIC_WORD"
        assert result.is_full is False

    def test_repeated_element(self) -> None:
        result = compute_basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0, 0))
        )
        assert result.status == "NOT_A_BASIC_WORD"
        assert result.obstruction == "repeated_element"

    def test_infeasible_prefix(self) -> None:
        # Path greedoid ab-bc: word (0, 1) is fine; word (1, 0) is also fine
        # because {bc} and {bc, ab} are both feasible. Try a foreign element.
        result = compute_basic_word_profile(
            BasicWordProfileRequest(system=_path_greedoid(), word=(5,))
        )
        assert result.status == "NOT_A_BASIC_WORD"
        assert result.obstruction == "foreign_element"


# ---------------------------------------------------------------------------
# Convex geometry
# ---------------------------------------------------------------------------


class TestConvexGeometry:
    def test_closed_family_has_top_and_bottom(self) -> None:
        result = compute_convex_geometry(
            ConvexGeometryRequest(system=_two_element_antimatroid())
        )
        empty = ()
        full = (0, 1)
        assert empty in result.closed_family
        assert full in result.closed_family

    def test_complement_map_inverse(self) -> None:
        result = compute_convex_geometry(
            ConvexGeometryRequest(system=_two_element_antimatroid())
        )
        # The complement map reverses inclusion: empty feasible -> full closed.
        lookup = dict(result.complement_map)
        assert lookup[()] == (0, 1)
        assert lookup[(0, 1)] == ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_unique_ground_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            FiniteFeasibleSetSystem(ground=("a", "a"), feasible=((),))

    def test_unsorted_feasible_set_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sorted"):
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((1, 0),))

    def test_duplicate_feasible_set_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate-free"):
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((), (0,), (0,)))

    def test_index_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError, match="out of range"):
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((0, 5),))


# ---------------------------------------------------------------------------
# Native helpers
# ---------------------------------------------------------------------------


def test_union_closed_true_for_antimatroid() -> None:
    assert greedoids.union_closed(_two_element_antimatroid())


def test_feasible_continuations() -> None:
    cont = greedoids.feasible_continuations(_two_element_antimatroid(), frozenset({0}))
    assert set(cont) == {1}
