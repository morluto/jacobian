"""Exact contract tests for relational embedding search."""

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
    search_embedding,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures._models import (
    EmbeddingSearchRequest,
    EmbeddingSearchResult,
    InducedEmbeddingCheckResult,
)

OPERATION_ID = "relational.embedding.search.compute"

_EDGE = (FiniteRelationSymbol(symbol_id="E", arity=2),)


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


def _bare_pair() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, ((),))


def _edge_plus_isolate() -> FiniteRelationalStructure:
    return _structure(3, _EDGE, (((0, 1),),))


class TestEmbeddingKnownAnswers:
    def test_bare_pair_skips_constant_map(self) -> None:
        """Plain search returns the constant (0, 0); embedding search
        must reject (0, 1), whose image contains an absent source edge."""
        plain = search_homomorphism(_bare_pair(), _edge_plus_isolate())
        assert plain.check is not None
        assert plain.check.carrier_map == (0, 0)
        result = search_embedding(_bare_pair(), _edge_plus_isolate())
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == (0, 2)
        assert (result.candidates_examined, result.total_candidates) == (3, 9)
        assert (
            result.check.induced_invariant
            == "EVERY_SOURCE_RELATION_TUPLE_REFLECTS_INTO_THE_TARGET"
        )

    def test_edge_into_loop_is_exhausted(self) -> None:
        """The edge homomorphically maps onto the loop point, but no
        injective map exists on a one-vertex carrier."""
        loop = _structure(1, _EDGE, (((0, 0),),))
        plain = search_homomorphism(_directed_edge(), loop)
        assert plain.status is HomomorphismSearchStatus.FOUND
        result = search_embedding(_directed_edge(), loop)
        assert result.status is HomomorphismSearchStatus.EXHAUSTED
        assert result.check is None
        assert (result.candidates_examined, result.total_candidates) == (1, 1)

    def test_three_cycle_first_embedding(self) -> None:
        result = search_embedding(_three_cycle(), _three_cycle())
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == (0, 1, 2)
        assert (result.candidates_examined, result.total_candidates) == (6, 27)

    def test_empty_source_embeds(self) -> None:
        source = _structure(0, _EDGE, ((),))
        result = search_embedding(source, _three_cycle())
        assert result.status is HomomorphismSearchStatus.FOUND
        assert result.check is not None
        assert result.check.carrier_map == ()
        assert (result.candidates_examined, result.total_candidates) == (1, 1)

    def test_larger_source_than_target_is_exhausted(self) -> None:
        """Four bare vertices cannot inject into three: pigeonhole
        exhaustion over the complete 3^4 space."""
        source = _structure(4, _EDGE, ((),))
        target = _structure(3, _EDGE, ((),))
        result = search_embedding(source, target)
        assert result.status is HomomorphismSearchStatus.EXHAUSTED
        assert (result.candidates_examined, result.total_candidates) == (81, 81)


class TestEmbeddingDefiningInvariant:
    def test_found_map_is_an_induced_embedding(self) -> None:
        source, target = _bare_pair(), _edge_plus_isolate()
        result = search_embedding(source, target)
        assert result.check is not None
        assert len(set(result.check.carrier_map)) == source.carrier_size
        assert all(
            p.matching_cells == p.relation_cells
            for p in result.check.reflection_profiles
        )
        assert (
            check_homomorphism(source, target, result.check.carrier_map).status
            is HomomorphismStatus.HOMOMORPHISM
        )

    def test_found_map_is_lexicographically_first_embedding(self) -> None:
        """An independent replay over distinct-image maps agrees the
        retained map is first."""
        from itertools import product

        source, target = _three_cycle(), _three_cycle()
        result = search_embedding(source, target)
        assert result.check is not None
        first = next(
            candidate
            for candidate in product(range(3), repeat=3)
            if len(set(candidate)) == 3
            and check_homomorphism(source, target, candidate).status
            is HomomorphismStatus.HOMOMORPHISM
        )
        assert result.check.carrier_map == first

    def test_embedding_found_implies_homomorphism_found(self) -> None:
        pairs = (
            (_bare_pair(), _edge_plus_isolate()),
            (_directed_edge(), _structure(1, _EDGE, (((0, 0),),))),
            (_three_cycle(), _three_cycle()),
        )
        for source, target in pairs:
            embedded = search_embedding(source, target)
            plain = search_homomorphism(source, target)
            assert (embedded.status is HomomorphismSearchStatus.FOUND) <= (
                plain.status is HomomorphismSearchStatus.FOUND
            )


class TestEmbeddingAdmission:
    def test_signature_mismatch_is_typed_rejected_on_both_paths(self) -> None:
        nullary = (FiniteRelationSymbol(symbol_id="N", arity=0),)
        source = _directed_edge()
        target = _structure(2, nullary, (((),),))
        with pytest.raises(OperationDomainValidationError) as native_info:
            search_embedding(source, target)
        assert (
            native_info.value.errors()[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )
        payload = {
            "source": json.loads(source.model_dump_json()),
            "target": json.loads(target.model_dump_json()),
        }
        with pytest.raises(ValidationError) as wire_info:
            EmbeddingSearchRequest.model_validate(payload)
        assert (
            wire_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.signature_mismatch"
        )

    def test_non_structures_are_typed_rejected_natively(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            search_embedding("edge", _three_cycle())  # type: ignore[arg-type]

    def test_oversized_space_is_resource_refused(self) -> None:
        source = _structure(17, (), ())
        target = _structure(2, (), ())
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            search_embedding(source, target)
        assert exc_info.value.errors()[0]["type"] == (
            "relational.homomorphism.search_space"
        )


class TestEmbeddingComposition:
    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.logic.relational_structures._tools import (
            _embedding_search,
        )

        source, target = _bare_pair(), _edge_plus_isolate()
        request = EmbeddingSearchRequest(source=source, target=target)
        assert _embedding_search(request) == search_embedding(source, target)

    def test_serialization_round_trips(self) -> None:
        found = search_embedding(_bare_pair(), _edge_plus_isolate())
        assert (
            EmbeddingSearchResult.model_validate_json(found.model_dump_json()) == found
        )
        loop = _structure(1, _EDGE, (((0, 0),),))
        exhausted = search_embedding(_directed_edge(), loop)
        assert (
            EmbeddingSearchResult.model_validate_json(exhausted.model_dump_json())
            == exhausted
        )

    def test_plain_homomorphism_claim_cannot_be_used_as_embedding(self) -> None:
        source, target = _bare_pair(), _edge_plus_isolate()
        plain = check_homomorphism(source, target, (0, 1))
        assert plain.status is HomomorphismStatus.HOMOMORPHISM
        payload = json.loads(search_embedding(source, target).model_dump_json())
        payload["check"] = json.loads(plain.model_dump_json())
        with pytest.raises(ValidationError):
            EmbeddingSearchResult.model_validate_json(json.dumps(payload))

    def test_forged_noninjective_found_is_rejected(self) -> None:
        result = search_embedding(_bare_pair(), _edge_plus_isolate())
        payload = json.loads(result.model_dump_json())
        payload["check"]["carrier_map"] = [0, 0]
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSearchResult.model_validate_json(json.dumps(payload))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.carrier_map_not_injective"
        )

    def test_canonical_induced_check_rejects_noninjective_map(self) -> None:
        result = search_embedding(_bare_pair(), _edge_plus_isolate())
        assert result.check is not None
        payload = json.loads(result.check.model_dump_json())
        payload["carrier_map"] = [0, 0]
        with pytest.raises(ValidationError) as exc_info:
            InducedEmbeddingCheckResult.model_validate_json(json.dumps(payload))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.carrier_map_not_injective"
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
        native = search_embedding(payload.source, payload.target)
        assert tool.run(payload) == native
        assert native.status is HomomorphismSearchStatus.FOUND

    def test_operation_is_published_in_the_catalog(self) -> None:
        operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in operation_ids
