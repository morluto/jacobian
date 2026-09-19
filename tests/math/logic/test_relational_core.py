"""Exact contract tests for relational core computation."""

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
    HomomorphismStatus,
    check_homomorphism,
    compute_core,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCoreRequest,
    HomomorphismCoreResult,
)

OPERATION_ID = "relational.core.compute"

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


def _v_shape() -> FiniteRelationalStructure:
    return _structure(3, _EDGE, (((0, 1), (0, 2)),))


def _directed_edge() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1),),))


class TestCoreKnownAnswers:
    def test_v_shape_core_is_single_edge(self) -> None:
        result = compute_core(_v_shape())
        assert result.inclusion == (0, 1)
        assert result.retraction == (0, 1, 1)
        assert result.core.carrier_size == 2
        assert result.core.relation_tables == (((0, 1),),)

    def test_three_cycle_is_rigid(self) -> None:
        result = compute_core(_three_cycle())
        assert result.inclusion == (0, 1, 2)
        assert result.retraction == (0, 1, 2)
        assert result.core == _three_cycle()

    def test_empty_signature_collapses_to_point(self) -> None:
        source = _structure(2, (), ())
        result = compute_core(source)
        assert result.inclusion == (0,)
        assert result.retraction == (0, 0)
        assert result.core.carrier_size == 1

    def test_loop_point_is_rigid(self) -> None:
        source = _structure(1, _EDGE, (((0, 0),),))
        result = compute_core(source)
        assert result.inclusion == (0,)
        assert result.retraction == (0,)
        assert result.core == source

    def test_empty_structure_is_its_own_core(self) -> None:
        source = _structure(0, _EDGE, ((),))
        result = compute_core(source)
        assert result.inclusion == ()
        assert result.retraction == ()
        assert result.core == source


class TestCoreDefiningInvariant:
    def test_retraction_is_idempotent_endomorphism(self) -> None:
        """inclusion ∘ retraction replays HOMOMORPHISM through the check
        operation on every fixture."""
        for source in (_v_shape(), _three_cycle(), _directed_edge()):
            result = compute_core(source)
            composed = tuple(result.inclusion[label] for label in result.retraction)
            check = check_homomorphism(source, source, composed)
            assert check.status is HomomorphismStatus.HOMOMORPHISM
            assert tuple(sorted(set(composed))) == result.inclusion

    def test_mixed_retraction_is_idempotent_and_serializes(self) -> None:
        """A core reached through a permuting level keeps a genuine
        retraction: the composed map is idempotent, the section identity
        holds, and the result round-trips through its own decoder."""
        source = _structure(3, _EDGE, (((0, 1), (1, 2), (2, 1)),))
        result = compute_core(source)
        assert result.core == _structure(2, _EDGE, (((0, 1), (1, 0)),))
        assert result.inclusion == (1, 2)
        assert all(
            result.retraction[result.inclusion[core_label]] == core_label
            for core_label in range(len(result.inclusion))
        )
        composed = tuple(result.inclusion[label] for label in result.retraction)
        assert all(composed[composed[label]] == composed[label] for label in range(3))
        assert (
            HomomorphismCoreResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_core_tables_are_induced(self) -> None:
        """Every core tuple lifts to a source tuple on the inclusion
        image and every covered source tuple descends."""
        result = compute_core(_v_shape())
        image = set(result.inclusion)
        rank = {label: position for position, label in enumerate(result.inclusion)}
        for symbol_index, table in enumerate(_v_shape().relation_tables):
            expected = tuple(
                sorted(
                    {
                        tuple(rank[coordinate] for coordinate in row)
                        for row in table
                        if all(coordinate in image for coordinate in row)
                    }
                )
            )
            assert result.core.relation_tables[symbol_index] == expected

    def test_core_is_minimal(self) -> None:
        """Every endomorphism of a computed core is surjective:
        independent enumeration through the count operation."""
        from itertools import product

        for source in (_v_shape(), _three_cycle(), _directed_edge()):
            core = compute_core(source).core
            size = core.carrier_size
            for candidate in product(range(size), repeat=size):
                check = check_homomorphism(core, core, candidate)
                if check.status is HomomorphismStatus.HOMOMORPHISM:
                    assert len(set(candidate)) == size

    def test_core_is_homomorphically_equivalent(self) -> None:
        """Retraction preserves homomorphism existence (not counts: the
        V shape admits 2 edge maps collapsing onto 1 core map)."""
        from jacobian.math.logic.relational_structures import search_homomorphism

        loop_point = _structure(1, _EDGE, (((0, 0),),))
        bare_point = _structure(1, _EDGE, ((),))
        probes = (_directed_edge(), _three_cycle(), loop_point, bare_point)
        for source in (_v_shape(), _three_cycle()):
            core = compute_core(source).core
            for probe in probes:
                assert search_homomorphism(probe, source).status == (
                    search_homomorphism(probe, core).status
                )


class TestCoreAdmission:
    def test_non_structure_is_typed_rejected_natively(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_core("edge")  # type: ignore[arg-type]

    def test_oversized_carrier_is_resource_refused(self) -> None:
        source = _structure(8, (), ())
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            compute_core(source)
        assert exc_info.value.errors()[0]["type"] == ("relational.core.search_work")

    def test_six_point_empty_signature_is_accepted(self) -> None:
        source = _structure(6, (), ())
        result = compute_core(source)
        assert result.core.carrier_size == 1
        assert result.inclusion == (0,)
        assert result.retraction == (0,) * 6


class TestCoreComposition:
    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.logic.relational_structures._tools import (
            _homomorphism_core,
        )

        source = _v_shape()
        request = HomomorphismCoreRequest(source=source)
        assert _homomorphism_core(request) == compute_core(source)

    def test_serialization_round_trips(self) -> None:
        result = compute_core(_v_shape())
        assert (
            HomomorphismCoreResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_forged_section_violation_is_rejected(self) -> None:
        result = compute_core(_v_shape())
        payload = json.loads(result.model_dump_json())
        payload["retraction"] = [0, 0, 1]
        with pytest.raises(ValidationError) as exc_info:
            HomomorphismCoreResult.model_validate_json(json.dumps(payload))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "relational.homomorphism.section_identity"
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
        native = compute_core(payload.source)
        assert tool.run(payload) == native
        assert native.inclusion == (0, 1)

    def test_operation_is_published_in_the_catalog(self) -> None:
        operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in operation_ids
