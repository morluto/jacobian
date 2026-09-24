"""Independent exact contract tests for finite relational reducts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    RelationalReductRequest,
    RelationalReductResult,
    check_homomorphism,
    reduct_structure,
)
from jacobian.math.logic.relational_structures._admission import (
    MAX_RELATIONAL_REDUCT_WORK,
    admit_relational_reduct,
)
from jacobian.math.logic.relational_structures._tools import TOOLS


def _source() -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=3,
        signature=(
            FiniteRelationSymbol(symbol_id="E", arity=2),
            FiniteRelationSymbol(symbol_id="P", arity=1),
            FiniteRelationSymbol(symbol_id="True0", arity=0),
        ),
        relation_tables=(
            ((0, 1), (1, 2)),
            ((1,), (2,)),
            ((),),
        ),
    )


def test_reduct_uses_source_order_even_when_selection_is_reversed() -> None:
    source = _source()

    result = reduct_structure(source, ("True0", "E"))

    assert result.source == source
    assert result.source_symbol_indices == (0, 2)
    assert result.reduct == FiniteRelationalStructure(
        carrier_size=3,
        signature=(source.signature[0], source.signature[2]),
        relation_tables=(source.relation_tables[0], source.relation_tables[2]),
    )
    identity = check_homomorphism(
        result.reduct, result.reduct, tuple(range(result.reduct.carrier_size))
    )
    assert identity.status is HomomorphismStatus.HOMOMORPHISM


def test_empty_reduct_keeps_empty_carrier_and_has_empty_signature() -> None:
    source = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="True0", arity=0),),
        relation_tables=(((),),),
    )

    result = reduct_structure(source, ())

    assert result.source_symbol_indices == ()
    assert result.reduct == FiniteRelationalStructure(carrier_size=0)


def test_result_serialization_keeps_symbol_transport() -> None:
    result = reduct_structure(_source(), ("P", "E"))

    restored = RelationalReductResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.source_symbol_indices == (0, 1)


def test_reduct_admission_prices_rows_and_coordinates_before_construction() -> None:
    indices, work, output_bound = admit_relational_reduct(_source(), ("E", "P"))

    assert indices == (0, 1)
    assert work == (1 + 2 * 3) + (1 + 2 * 2)
    source_bytes = len(_source().model_dump_json().encode("utf-8"))
    assert output_bound == 2 * source_bytes + 256 + 12 * len(indices)
    assert work <= MAX_RELATIONAL_REDUCT_WORK


def test_complete_result_size_bound_is_enforced_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.logic.relational_structures import _admission

    source = _source()
    _, _, bound = admit_relational_reduct(source, ("P",))

    class Limits:
        max_output_bytes = bound

    monkeypatch.setattr(_admission, "CanonicalLimits", Limits)
    assert admit_relational_reduct(source, ("P",))[2] == bound

    Limits.max_output_bytes = bound - 1
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        admit_relational_reduct(source, ("P",))
    assert exc_info.value.errors()[0]["type"] == "relational.reduct.output_bound"


def test_reduct_rejects_unknown_or_repeated_symbols() -> None:
    with pytest.raises(ValidationError) as unknown:
        RelationalReductRequest(source=_source(), symbol_ids=("missing",))
    assert (
        unknown.value.errors()[0]["type"]
        == "relational.homomorphism.reduct.symbol_id_unknown"
    )

    with pytest.raises(ValidationError) as repeated:
        RelationalReductRequest(source=_source(), symbol_ids=("E", "E"))
    assert (
        repeated.value.errors()[0]["type"]
        == "relational.homomorphism.reduct.symbol_ids_not_unique"
    )


def test_native_admission_rejects_work_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.logic.relational_structures import _admission

    monkeypatch.setattr(_admission, "MAX_RELATIONAL_REDUCT_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        reduct_structure(_source(), ("E",))
    assert exc_info.value.errors()[0]["type"] == "relational.reduct.work_bound"


def test_native_kernel_rejects_validation_bypassed_unknown_symbol() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        reduct_structure(_source(), ("missing",))
    assert exc_info.value.errors()[0]["type"] == "relational.reduct.symbol_id_unknown"


def test_reduct_catalog_example_is_wired() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "relational_structure.reduct.compute"
    )
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)
    assert result.reduct.signature == (FiniteRelationSymbol(symbol_id="P", arity=1),)
    assert result.source_symbol_indices == (1,)
