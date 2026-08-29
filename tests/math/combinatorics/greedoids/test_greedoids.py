"""Tests for greedoid operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics import greedoids
from jacobian.math.combinatorics.greedoids import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.greedoids._models import (
    MAX_GROUND_LABEL_UTF8_BYTES,
    BasesRequest,
    BasesResult,
    BasicWordProfileRequest,
    ConvexGeometryRequest,
    ConvexGeometryResult,
    RankRequest,
    RankResult,
    RecognizeRequest,
)
from jacobian.math.combinatorics.greedoids._tools import (
    TOOLS,
    _bases,
    _basic_word_profile,
    _convex_geometry,
    _rank,
    _recognize,
)

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


@pytest.mark.parametrize(
    "result",
    (
        RankResult.model_construct(status="GREEDOID", rank=None),
        BasesResult.model_construct(status="NOT_A_GREEDOID", rank=2, bases=()),
        ConvexGeometryResult.model_construct(
            status="NOT_AN_ANTIMATROID", closed_family=((),), obstruction="x"
        ),
    ),
)
def test_result_models_reject_mixed_outcome_branches(
    result: RankResult | BasesResult | ConvexGeometryResult,
) -> None:
    with pytest.raises(ValidationError):
        result.model_validate(result.model_dump())


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


def test_recognition_rejects_a_ground_label_outside_utf8_budget() -> None:
    with pytest.raises(OperationDomainValidationError, match="ground label exceeds"):
        _recognize(
            RecognizeRequest(
                system=FiniteFeasibleSetSystem(
                    ground=("x" * (MAX_GROUND_LABEL_UTF8_BYTES + 1),),
                    feasible=((),),
                )
            )
        )


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


class TestRecognize:
    def test_empty_ground_with_its_empty_feasible_set_is_a_greedoid(self) -> None:
        result = _recognize(
            RecognizeRequest(system=FiniteFeasibleSetSystem(ground=(), feasible=((),)))
        )

        assert result.status == "GREEDOID"
        assert result.rank == 0
        assert result.bases == ((),)
        assert result.ground_size == 0

    def test_two_element_antimatroid_is_greedoid(self) -> None:
        result = _recognize(RecognizeRequest(system=_two_element_antimatroid()))
        assert result.status == "GREEDOID"
        assert result.rank == 2
        assert result.bases == ((0, 1),)

    def test_path_greedoid_is_greedoid(self) -> None:
        result = _recognize(RecognizeRequest(system=_path_greedoid()))
        assert result.status == "GREEDOID"
        assert result.rank == 2

    def test_missing_empty_set(self) -> None:
        result = _recognize(RecognizeRequest(system=_non_greedoid_missing_empty()))
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "missing_empty_set"

    def test_inaccessible_feasible_set(self) -> None:
        result = _recognize(RecognizeRequest(system=_non_greedoid_inaccessible()))
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "inaccessible_feasible_set"
        assert result.feasible_set == (0, 1)

    def test_exchange_violation(self) -> None:
        result = _recognize(RecognizeRequest(system=_non_greedoid_exchange()))
        assert result.status == "NOT_A_GREEDOID"
        assert result.obstruction == "exchange_violation"


# ---------------------------------------------------------------------------
# Rank and bases
# ---------------------------------------------------------------------------


class TestRankAndBases:
    def test_rank_of_full_ground(self) -> None:
        result = _rank(RankRequest(system=_two_element_antimatroid()))
        assert result.rank == 2

    def test_rank_of_subset(self) -> None:
        # r({a}) = 1 because {a} is feasible and {a} is the largest feasible
        # subset of {a}.
        result = _rank(RankRequest(system=_two_element_antimatroid(), subset=(0,)))
        assert result.rank == 1

    def test_rank_of_empty_subset(self) -> None:
        result = _rank(RankRequest(system=_two_element_antimatroid(), subset=()))
        assert result.rank == 0

    def test_bases_of_full_ground(self) -> None:
        result = _bases(BasesRequest(system=_two_element_antimatroid()))
        assert result.rank == 2
        assert result.bases == ((0, 1),)

    def test_bases_of_subset(self) -> None:
        # Bases of {a} = {{a}} (rank 1).
        result = _bases(BasesRequest(system=_two_element_antimatroid(), subset=(0,)))
        assert result.rank == 1
        assert result.bases == ((0,),)

    def test_non_greedoid_cannot_claim_bases_or_rank(self) -> None:
        system = _non_greedoid_exchange()
        bases_result = _bases(BasesRequest(system=system))
        rank_result = _rank(RankRequest(system=system))
        assert bases_result.status == "NOT_A_GREEDOID"
        assert bases_result.bases == ()
        assert rank_result.status == "NOT_A_GREEDOID"
        assert rank_result.rank is None


# ---------------------------------------------------------------------------
# Basic word profile
# ---------------------------------------------------------------------------


class TestBasicWordProfile:
    def test_full_basic_word(self) -> None:
        result = _basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0, 1))
        )
        assert result.status == "BASIC_WORD"
        assert result.is_full is True
        assert result.rank == 2

    def test_prefix_basic_word(self) -> None:
        # Word (0,) is a basic word of length 1; not full.
        result = _basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0,))
        )
        assert result.status == "BASIC_WORD"
        assert result.is_full is False

    def test_repeated_element(self) -> None:
        result = _basic_word_profile(
            BasicWordProfileRequest(system=_two_element_antimatroid(), word=(0, 0))
        )
        assert result.status == "NOT_A_BASIC_WORD"
        assert result.obstruction == "repeated_element"

    def test_infeasible_prefix(self) -> None:
        # Path greedoid ab-bc: word (0, 1) is fine; word (1, 0) is also fine
        # because {bc} and {bc, ab} are both feasible. Try a foreign element.
        result = _basic_word_profile(
            BasicWordProfileRequest(system=_path_greedoid(), word=(5,))
        )
        assert result.status == "NOT_A_BASIC_WORD"
        assert result.obstruction == "foreign_element"


# ---------------------------------------------------------------------------
# Convex geometry
# ---------------------------------------------------------------------------


class TestConvexGeometry:
    def test_closed_family_has_top_and_bottom(self) -> None:
        result = _convex_geometry(
            ConvexGeometryRequest(system=_two_element_antimatroid())
        )
        empty = ()
        full = (0, 1)
        assert empty in result.closed_family
        assert full in result.closed_family

    def test_complement_map_inverse(self) -> None:
        result = _convex_geometry(
            ConvexGeometryRequest(system=_two_element_antimatroid())
        )
        # The complement map reverses inclusion: empty feasible -> full closed.
        lookup = dict(result.complement_map)
        assert lookup[()] == (0, 1)
        assert lookup[(0, 1)] == ()

    def test_non_antimatroid_cannot_claim_a_convex_geometry(self) -> None:
        result = _convex_geometry(
            ConvexGeometryRequest(
                system=FiniteFeasibleSetSystem(
                    ground=("a", "b"), feasible=((), (0,), (1,))
                )
            )
        )
        assert result.status == "NOT_AN_ANTIMATROID"
        assert result.closed_family == ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_unique_ground_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteFeasibleSetSystem(ground=("a", "a"), feasible=((),))
        assert error.value.errors()[0]["type"] == "greedoid.ground_duplicate"

    def test_unsorted_feasible_set_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((1, 0),))
        assert error.value.errors()[0]["type"] == "greedoid.feasible_row_unsorted"

    def test_duplicate_feasible_set_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((), (0,), (0,)))
        assert error.value.errors()[0]["type"] == "greedoid.feasible_family_duplicate"

    def test_index_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((0, 5),))
        assert error.value.errors()[0]["type"] == "greedoid.feasible_index_out_of_range"

    def test_every_request_bounds_ground_size(self) -> None:
        # The carrier is structural only; each greedoid request owns the
        # execution-envelope ceiling on ground cardinality.
        system = FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(65)),
            feasible=((),),
        )
        assert isinstance(system, FiniteFeasibleSetSystem)

        operations = (
            lambda s: _recognize(RecognizeRequest(system=s)),
            lambda s: _rank(RankRequest(system=s)),
            lambda s: _bases(BasesRequest(system=s)),
            lambda s: _basic_word_profile(BasicWordProfileRequest(system=s, word=(0,))),
            lambda s: _convex_geometry(ConvexGeometryRequest(system=s)),
        )
        for operation in operations:
            with pytest.raises(OperationDomainValidationError) as error:
                operation(system)
            assert (
                error.value.errors()[0]["type"] == "greedoid.ground_size_exceeds_budget"
            )

    def test_recognize_request_bounds_feasible_set_count(self) -> None:
        feasible = []
        for mask in range(1, 4098):
            feasible.append(tuple(i for i in range(13) if (mask >> i) & 1))
        system = FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(13)),
            feasible=tuple(feasible),
        )

        with pytest.raises(OperationDomainValidationError) as error:
            _recognize(RecognizeRequest(system=system))
        assert (
            error.value.errors()[0]["type"] == "greedoid.feasible_count_exceeds_budget"
        )


# ---------------------------------------------------------------------------
# Native helpers
# ---------------------------------------------------------------------------


def test_union_closed_true_for_antimatroid() -> None:
    assert greedoids.union_closed(_two_element_antimatroid())


def test_native_convex_geometry_returns_the_canonical_result() -> None:
    result = greedoids.antimatroid_to_convex_geometry(_two_element_antimatroid())

    assert result.status == "ANTIMATROID"
    assert result.closed_family == ((0, 1), (1,), (0,), ())


def test_feasible_continuations() -> None:
    cont = greedoids.feasible_continuations(_two_element_antimatroid(), frozenset({0}))
    assert set(cont) == {1}


def test_native_greedoid_consumers_reject_an_unrecognized_family() -> None:
    """Structural feasible-set data cannot claim greedoid-derived facts."""
    system = _non_greedoid_exchange()
    calls = (
        lambda: greedoids.rank(system),
        lambda: greedoids.bases(system),
        lambda: greedoids.feasible_continuations(system, frozenset()),
        lambda: greedoids.basic_word_profile(system, ()),
    )
    for call in calls:
        with pytest.raises(ValueError, match="recognized greedoid"):
            call()


def test_native_convex_geometry_consumer_rejects_a_non_antimatroid() -> None:
    system = FiniteFeasibleSetSystem(ground=("a", "b"), feasible=((), (0,), (1,)))
    with pytest.raises(ValueError, match="full support and union closure"):
        greedoids.antimatroid_to_convex_geometry(system)


# ---------------------------------------------------------------------------
# Native carrier admission
# ---------------------------------------------------------------------------


class TestNativeCarrierAdmission:
    """Native entry points enforce the same envelope as their requests."""

    def _over_row_budget_system(self) -> FiniteFeasibleSetSystem:
        feasible = []
        for mask in range(1, 4098):
            feasible.append(tuple(i for i in range(13) if (mask >> i) & 1))
        return FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(13)),
            feasible=tuple(feasible),
        )

    def test_recognize_rejects_family_over_row_budget(self) -> None:
        system = self._over_row_budget_system()
        assert isinstance(system, FiniteFeasibleSetSystem)
        with pytest.raises(ValueError, match="feasible-set count"):
            greedoids.recognize(system)

    def test_union_closed_rejects_family_over_row_budget(self) -> None:
        with pytest.raises(ValueError, match="feasible-set count"):
            greedoids.union_closed(self._over_row_budget_system())

    def test_native_entries_reject_ground_over_budget(self) -> None:
        system = FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(65)),
            feasible=((),),
        )
        calls = (
            lambda s: greedoids.recognize(s),
            lambda s: greedoids.union_closed(s),
            lambda s: greedoids.rank(s),
            lambda s: greedoids.bases(s),
            lambda s: greedoids.feasible_continuations(s, frozenset()),
            lambda s: greedoids.basic_word_profile(s, ()),
            lambda s: greedoids.antimatroid_to_convex_geometry(s),
        )
        for call in calls:
            with pytest.raises(ValueError, match="ground size"):
                call(system)

    def test_boundary_carrier_still_recognized(self) -> None:
        system = FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(64)),
            feasible=((),),
        )
        result = greedoids.recognize(system)
        assert result.status == "GREEDOID"
        assert result.rank == 0
