"""Exact contract tests for bounded relational homomorphism counting."""

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
    HomomorphismSearchStatus,
    HomomorphismStatus,
    check_homomorphism,
    count_homomorphisms,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCountRequest,
    HomomorphismCountResult,
)

OPERATION_ID = "relational.homomorphism.count.compute"

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


def _two_clique() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1), (1, 0)),))


class TestCountKnownAnswers:
    @pytest.mark.parametrize(
        ("source", "target", "count", "total"),
        (
            ("edge", "cycle", 3, 9),
            ("cycle", "edge", 0, 8),
            ("clique", "clique", 2, 4),
            ("loop", "loop", 1, 1),
        ),
    )
    def test_exact_counts(
        self, source: str, target: str, count: int, total: int
    ) -> None:
        structures = {
            "edge": _directed_edge(),
            "cycle": _three_cycle(),
            "clique": _two_clique(),
            "loop": _structure(1, _EDGE, (((0, 0),),)),
        }
        result = count_homomorphisms(structures[source], structures[target])
        assert (result.count, result.total_candidates) == (count, total)

    def test_empty_signature_counts_every_map(self) -> None:
        source = _structure(2, (), ())
        target = _structure(3, (), ())
        result = count_homomorphisms(source, target)
        assert (result.count, result.total_candidates) == (9, 9)

    def test_empty_source_counts_one(self) -> None:
        source = _structure(0, _EDGE, ((),))
        result = count_homomorphisms(source, _three_cycle())
        assert (result.count, result.total_candidates) == (1, 1)

    def test_empty_carrier_counts_zero(self) -> None:
        source = _directed_edge()
        target = _structure(0, _EDGE, ((),))
        result = count_homomorphisms(source, target)
        assert (result.count, result.total_candidates) == (0, 0)

    def test_false_nullary_target_counts_zero(self) -> None:
        source = _structure(1, _NULLARY, ((((),),)))
        target = _structure(1, _NULLARY, ((),))
        result = count_homomorphisms(source, target)
        assert (result.count, result.total_candidates) == (0, 1)


class TestCountSearchAgreement:
    @pytest.mark.parametrize(
        ("source", "target"),
        (
            ("edge", "cycle"),
            ("cycle", "edge"),
            ("clique", "clique"),
            ("loop", "loop"),
        ),
    )
    def test_search_found_iff_count_positive(self, source: str, target: str) -> None:
        """First-witness search and complete-profile count agree on
        existence across satisfiable and unsatisfiable pairs."""
        structures = {
            "edge": _directed_edge(),
            "cycle": _three_cycle(),
            "clique": _two_clique(),
            "loop": _structure(1, _EDGE, (((0, 0),),)),
        }
        found = search_homomorphism(structures[source], structures[target])
        counted = count_homomorphisms(structures[source], structures[target])
        assert (found.status is HomomorphismSearchStatus.FOUND) == (counted.count >= 1)
        assert found.total_candidates == counted.total_candidates

    def test_count_matches_independent_replay(self) -> None:
        """An independent loop over every map through the check operation
        reproduces the count."""
        from itertools import product

        source, target = _directed_edge(), _three_cycle()
        expected = sum(
            1
            for candidate in product(range(3), repeat=2)
            if check_homomorphism(source, target, candidate).status
            is HomomorphismStatus.HOMOMORPHISM
        )
        assert count_homomorphisms(source, target).count == expected == 3


class TestCountAdmission:
    def test_signature_mismatch_is_typed_rejected_on_both_paths(self) -> None:
        source = _directed_edge()
        target = _structure(2, _NULLARY, (((),),))
        with pytest.raises(OperationDomainValidationError) as native_info:
            count_homomorphisms(source, target)
        assert (
            native_info.value.errors()[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )
        payload = {
            "source": json.loads(source.model_dump_json()),
            "target": json.loads(target.model_dump_json()),
        }
        with pytest.raises(ValidationError) as wire_info:
            HomomorphismCountRequest.model_validate(payload)
        assert (
            wire_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )

    def test_non_structures_are_typed_rejected_natively(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            count_homomorphisms("edge", _three_cycle())  # type: ignore[arg-type]

    def test_oversized_space_is_resource_refused(self) -> None:
        source = _structure(17, (), ())
        target = _structure(2, (), ())
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            count_homomorphisms(source, target)
        assert exc_info.value.errors()[0]["type"] == (
            "relational.homomorphism.search_space"
        )


class TestCountComposition:
    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.logic.relational_structures._tools import (
            _homomorphism_count,
        )

        source, target = _directed_edge(), _three_cycle()
        request = HomomorphismCountRequest(source=source, target=target)
        assert _homomorphism_count(request) == count_homomorphisms(source, target)

    def test_serialization_round_trips(self) -> None:
        result = count_homomorphisms(_directed_edge(), _three_cycle())
        assert (
            HomomorphismCountResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_forged_count_above_total_is_rejected(self) -> None:
        result = count_homomorphisms(_directed_edge(), _three_cycle())
        payload = json.loads(result.model_dump_json())
        payload["count"] = 10
        with pytest.raises(ValidationError) as exc_info:
            HomomorphismCountResult.model_validate_json(json.dumps(payload))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.count_range"
        )

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
        native = count_homomorphisms(payload.source, payload.target)
        assert tool.run(payload) == native
        assert native.count == 3

    def test_operation_is_published_in_the_catalog(self) -> None:
        operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in operation_ids
