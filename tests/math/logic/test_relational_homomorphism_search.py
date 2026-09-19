"""Exact contract tests for bounded relational homomorphism search."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismSearchResult,
    HomomorphismSearchStatus,
    HomomorphismStatus,
    check_homomorphism,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures._admission import (
    MAX_SEARCH_CANDIDATES,
    MAX_SEARCH_TUPLE_REPLAYS,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismSearchRequest,
)

OPERATION_ID = "relational.homomorphism.search.compute"

_EDGE = (FiniteRelationSymbol(symbol_id="E", arity=2),)
_NULLARY = (FiniteRelationSymbol(symbol_id="N", arity=0),)


def _structure(
    carrier_size: int,
    signature: tuple[FiniteRelationSymbol, ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size, signature=signature, relation_tables=tables
    )


def _three_cycle() -> FiniteRelationalStructure:
    return _structure(3, _EDGE, (((0, 1), (1, 2), (2, 0)),))


def _directed_edge() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1),),))


def _single_directed_edge() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1),),))


class TestSearchKnownAnswers:
    def test_edge_into_three_cycle_finds_first_map(self) -> None:
        result = search_homomorphism(_directed_edge(), _three_cycle())
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == (0, 1)
        assert result.check.status is HomomorphismStatus.HOMOMORPHISM
        assert (result.candidates_examined, result.total_candidates) == (2, 9)

    def test_three_cycle_into_single_edge_is_exhausted(self) -> None:
        result = search_homomorphism(_three_cycle(), _single_directed_edge())
        assert result.status is HomomorphismSearchStatus.EXHAUSTED
        assert result.check is None
        assert (result.candidates_examined, result.total_candidates) == (8, 8)

    def test_empty_signature_finds_all_zero_map_first(self) -> None:
        source = _structure(2, (), ())
        target = _structure(3, (), ())
        result = search_homomorphism(source, target)
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == (0, 0)
        assert (result.candidates_examined, result.total_candidates) == (1, 9)

    def test_empty_source_finds_empty_map(self) -> None:
        source = _structure(0, _EDGE, ((),))
        target = _three_cycle()
        result = search_homomorphism(source, target)
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == ()
        assert (result.candidates_examined, result.total_candidates) == (1, 1)

    def test_nonempty_source_into_empty_carrier_is_exhausted(self) -> None:
        source = _directed_edge()
        target = _structure(0, _EDGE, ((),))
        result = search_homomorphism(source, target)
        assert result.status is HomomorphismSearchStatus.EXHAUSTED
        assert (result.candidates_examined, result.total_candidates) == (0, 0)

    def test_false_nullary_target_exhausts_despite_map(self) -> None:
        """A carrier map exists but no homomorphism does: the source
        nullary truth has nowhere to transport."""
        source = _structure(1, _NULLARY, ((((),),)))
        target = _structure(1, _NULLARY, ((),))
        result = search_homomorphism(source, target)
        assert result.status is HomomorphismSearchStatus.EXHAUSTED
        assert (result.candidates_examined, result.total_candidates) == (1, 1)

    def test_found_map_replays_through_the_check(self) -> None:
        """The retained FOUND witness is the check operation's own verdict
        on the same map: producer-consumer composition."""
        source, target = _directed_edge(), _three_cycle()
        result = search_homomorphism(source, target)
        assert result.check is not None
        assert (
            check_homomorphism(source, target, result.check.carrier_map) == result.check
        )

    def test_found_map_is_lexicographically_first(self) -> None:
        """An independent replay over every map agrees the retained map is
        the first transporting one."""
        from itertools import product

        source, target = _directed_edge(), _three_cycle()
        result = search_homomorphism(source, target)
        assert result.check is not None
        first = next(
            candidate
            for candidate in product(range(3), repeat=2)
            if check_homomorphism(source, target, candidate).status
            is HomomorphismStatus.HOMOMORPHISM
        )
        assert result.check.carrier_map == first


class TestSearchAdmission:
    def test_signature_mismatch_is_typed_rejected_on_both_paths(self) -> None:
        source = _directed_edge()
        target = _structure(2, _NULLARY, (((),),))
        with pytest.raises(OperationDomainValidationError) as native_info:
            search_homomorphism(source, target)
        assert (
            native_info.value.errors()[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )
        payload = {
            "source": json.loads(source.model_dump_json()),
            "target": json.loads(target.model_dump_json()),
        }
        with pytest.raises(ValidationError) as wire_info:
            HomomorphismSearchRequest.model_validate(payload)
        assert (
            wire_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )

    def test_non_structures_are_typed_rejected_natively(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            search_homomorphism("edge", _three_cycle())  # type: ignore[arg-type]

    def test_oversized_space_is_resource_refused(self) -> None:
        source = _structure(17, (), ())
        target = _structure(2, (), ())
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            search_homomorphism(source, target)
        assert exc_info.value.errors()[0]["type"] == (
            "relational.homomorphism.search_space"
        )

    def test_space_boundary_is_accepted(self) -> None:
        source = _structure(16, (), ())
        target = _structure(2, (), ())
        result = search_homomorphism(source, target)
        assert result.total_candidates == MAX_SEARCH_CANDIDATES
        assert result.status is HomomorphismSearchStatus.FOUND

    def test_joint_work_over_budget_is_resource_refused(self) -> None:
        dense_table = tuple(
            (a, b, c, d)
            for a in range(8)
            for b in range(8)
            for c in range(8)
            for d in range(8)
        )
        quaternary = tuple(
            FiniteRelationSymbol(symbol_id=f"R{i}", arity=4) for i in range(4)
        )
        source = _structure(8, quaternary, tuple(dense_table for _ in range(4)))
        target = _structure(
            2,
            quaternary,
            tuple(tuple((0, 0, 0, 0) for _ in range(16)) for _ in range(4)),
        )
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            search_homomorphism(source, target)
        assert exc_info.value.errors()[0]["type"] == (
            "relational.homomorphism.search_work"
        )

    def test_joint_work_boundary_is_accepted(self) -> None:
        dense_table = tuple(
            (a, b, c, d)
            for a in range(8)
            for b in range(8)
            for c in range(8)
            for d in range(8)
        )
        quaternary = (FiniteRelationSymbol(symbol_id="R", arity=4),)
        source = _structure(8, quaternary, (dense_table,))
        full_binary_table = tuple(
            (a, b, c, d)
            for a in range(2)
            for b in range(2)
            for c in range(2)
            for d in range(2)
        )
        target = _structure(2, quaternary, (full_binary_table,))
        result = search_homomorphism(source, target)
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.total_candidates * 4096 == MAX_SEARCH_TUPLE_REPLAYS


class TestSearchComposition:
    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.logic.relational_structures._tools import (
            _homomorphism_search,
        )

        source, target = _directed_edge(), _three_cycle()
        request = HomomorphismSearchRequest(source=source, target=target)
        assert _homomorphism_search(request) == search_homomorphism(source, target)

    def test_serialization_round_trips(self) -> None:
        found = search_homomorphism(_directed_edge(), _three_cycle())
        assert (
            HomomorphismSearchResult.model_validate_json(found.model_dump_json())
            == found
        )
        exhausted = search_homomorphism(_three_cycle(), _single_directed_edge())
        assert (
            HomomorphismSearchResult.model_validate_json(exhausted.model_dump_json())
            == exhausted
        )

    def test_forged_found_without_check_is_rejected(self) -> None:
        found = search_homomorphism(_directed_edge(), _three_cycle())
        payload = json.loads(found.model_dump_json())
        payload["check"] = None
        with pytest.raises(ValidationError):
            HomomorphismSearchResult.model_validate(payload)

    def test_forged_exhausted_with_unexamined_space_is_rejected(self) -> None:
        exhausted = search_homomorphism(_three_cycle(), _single_directed_edge())
        payload = json.loads(exhausted.model_dump_json())
        payload["candidates_examined"] = 0
        with pytest.raises(ValidationError):
            HomomorphismSearchResult.model_validate(payload)

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.logic.relational_structures._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == OPERATION_ID
        }
        assert set(tools) == {OPERATION_ID}
        tool = tools[OPERATION_ID]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        native = search_homomorphism(payload.source, payload.target)
        assert tool.run(payload) == native
        assert native.status is HomomorphismSearchStatus.FOUND

    def test_operation_is_published_in_the_catalog(self) -> None:
        operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in operation_ids
