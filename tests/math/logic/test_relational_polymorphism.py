"""Exact finite-operation preservation checks and small independent oracle."""

import json
from itertools import product

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
    RelationalPolymorphism,
    RelationalPolymorphismCheckResult,
    RelationalPolymorphismRequest,
    RelationalPolymorphismStatus,
    check_polymorphism,
    operations,
)
from jacobian.math.logic.relational_structures._admission import (
    MAX_POLYMORPHISM_COORDINATE_WORK,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_OPERATION_TABLE_CELLS,
)


def _binary_structure(mask: int) -> FiniteRelationalStructure:
    rows = tuple(product(range(2), repeat=2))
    return FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="R", arity=2),),
        relation_tables=(
            tuple(row for bit, row in enumerate(rows) if mask & (1 << bit)),
        ),
    )


def _candidate_tables(carrier_size: int, arity: int):
    domain = tuple(product(range(carrier_size), repeat=arity))
    for values in product(range(carrier_size), repeat=len(domain)):
        yield domain, values


def _oracle_preserves(structure, arity, domain, values) -> bool:
    operation = dict(zip(domain, values, strict=True))
    for relation in structure.relation_tables:
        for rows in product(relation, repeat=arity):
            output = tuple(
                operation[tuple(row[column] for row in rows)]
                for column in range(structure.signature[0].arity)
            )
            if output not in relation:
                return False
    return True


def test_exhaustive_all_binary_relations_and_binary_operations_match_oracle():
    for mask in range(16):
        structure = _binary_structure(mask)
        domain, _ = next(_candidate_tables(2, 2))
        for _domain, values in _candidate_tables(2, 2):
            expected = _oracle_preserves(structure, 2, domain, values)
            result = check_polymorphism(structure, 2, values)
            assert (
                result.status is RelationalPolymorphismStatus.POLYMORPHISM
            ) is expected
            if expected:
                assert isinstance(result.polymorphism, RelationalPolymorphism)
                assert result.witness is None
                assert result.polymorphism.operation_table == values
            else:
                assert result.polymorphism is None
                assert result.witness is not None
                relation = structure.relation_tables[0]
                assert all(row in relation for row in result.witness.input_rows)
                assert result.witness.output_row not in relation
                operation = dict(zip(domain, values, strict=True))
                assert result.witness.output_row == tuple(
                    operation[tuple(row[column] for row in result.witness.input_rows)]
                    for column in range(2)
                )


def test_projection_preserves_relations_and_nullary_truth_is_exact():
    binary_edge = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1),),),
    )
    projection = check_polymorphism(binary_edge, 2, (0, 0, 1, 1))
    assert projection.status is RelationalPolymorphismStatus.POLYMORPHISM
    assert projection.relation_profiles[0].input_combinations == 1

    false_nullary = FiniteRelationalStructure(
        carrier_size=1,
        signature=(FiniteRelationSymbol(symbol_id="F", arity=0),),
        relation_tables=((),),
    )
    true_nullary = FiniteRelationalStructure(
        carrier_size=1,
        signature=(FiniteRelationSymbol(symbol_id="T", arity=0),),
        relation_tables=(((),),),
    )

    assert check_polymorphism(false_nullary, 2, (0,)).status is (
        RelationalPolymorphismStatus.POLYMORPHISM
    )
    assert check_polymorphism(true_nullary, 2, (0,)).status is (
        RelationalPolymorphismStatus.POLYMORPHISM
    )


def test_empty_carrier_has_the_unique_empty_operation_table():
    source = FiniteRelationalStructure(carrier_size=0)
    result = check_polymorphism(source, 3, ())
    assert result.status is RelationalPolymorphismStatus.POLYMORPHISM
    assert result.polymorphism is not None
    assert result.polymorphism.operation_table == ()


def test_operation_table_axis_and_values_are_checked():
    source = FiniteRelationalStructure(carrier_size=2)
    with pytest.raises(ValidationError):
        RelationalPolymorphismRequest(source=source, arity=2, operation_table=(0, 1))
    with pytest.raises(ValidationError):
        RelationalPolymorphismRequest(source=source, arity=1, operation_table=(0, 2))
    with pytest.raises(ValidationError):
        RelationalPolymorphismRequest(
            source=source,
            arity=9,
            operation_table=(),
        )
    assert MAX_RELATIONAL_OPERATION_TABLE_CELLS == 16_384


def test_catalog_example_and_result_round_trip():
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "relational.polymorphism.check"
    )
    example = tool.examples[0]
    request = tool.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    result = tool.run(request)
    assert result.status is RelationalPolymorphismStatus.POLYMORPHISM
    assert result.polymorphism is not None
    restored = type(result).model_validate_json(result.model_dump_json(), strict=True)
    assert restored == result


def test_relation_product_bound_is_checked_before_tuple_expansion(monkeypatch):
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="R", arity=2),),
        relation_tables=(tuple(product(range(2), repeat=2)),),
    )

    def expansion_must_not_start(*_args, **_kwargs):
        raise AssertionError("relation products expanded before admission")

    monkeypatch.setattr(operations, "product", expansion_must_not_start)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        check_polymorphism(source, 8, (0,) * (2**8))
    assert exc_info.value.errors()[0]["type"] == (
        "relational.polymorphism.coordinate_work_bound"
    )
    assert MAX_POLYMORPHISM_COORDINATE_WORK < 4**8 * 2 * 8


def test_full_relation_product_bound_rejects_before_expansion(monkeypatch):
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=tuple(
            FiniteRelationSymbol(symbol_id=f"R{index}", arity=2) for index in range(2)
        ),
        relation_tables=(tuple(product(range(2), repeat=2)),) * 2,
    )

    def expansion_must_not_start(*_args, **_kwargs):
        raise AssertionError("relation products expanded before admission")

    monkeypatch.setattr(operations, "product", expansion_must_not_start)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        check_polymorphism(source, 8, (0,) * (2**8))
    assert exc_info.value.errors()[0]["type"] == (
        "relational.polymorphism.relation_product_bound"
    )


def test_native_kernel_rejects_malformed_operation_tables() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1),),),
    )
    with pytest.raises(OperationDomainValidationError) as axis:
        check_polymorphism(source, 2, (0, 1))
    assert axis.value.errors()[0]["type"] == "relational.polymorphism.table_axis"
    with pytest.raises(OperationDomainValidationError) as value:
        check_polymorphism(source, 1, (0, 2))
    assert value.value.errors()[0]["type"] == "relational.polymorphism.table_value"
    with pytest.raises(OperationDomainValidationError) as shape:
        check_polymorphism(source, 1, 0)
    assert shape.value.errors()[0]["type"] == "relational.polymorphism.table_shape"
    with pytest.raises(OperationDomainValidationError) as arity:
        check_polymorphism(source, True, (0, 1))
    assert arity.value.errors()[0]["type"] == "relational.polymorphism.arity"


def _witnessed_failure():
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1),),),
    )
    # f(0,0)=1 and 0 elsewhere: the single pair from R images to (1,0), not in R.
    result = check_polymorphism(source, 2, (1, 0, 0, 0))
    assert result.status is RelationalPolymorphismStatus.NOT_POLYMORPHISM
    assert result.witness is not None
    assert result.witness.input_rows == ((0, 1), (0, 1))
    assert result.witness.output_row == (1, 0)
    return result


def test_result_validation_is_structural_and_never_replays_the_kernel() -> None:
    result = _witnessed_failure()
    payload = json.loads(result.model_dump_json())

    restored = RelationalPolymorphismCheckResult.model_validate_json(
        json.dumps(payload)
    )
    assert restored == result

    # An authored witness whose output row is a well-shaped carrier tuple but
    # is not the coordinatewise table image stays structurally valid: replay
    # of the image belongs to the admitted check_polymorphism kernel only.
    payload["witness"]["output_row"] = [0, 0]
    non_image = RelationalPolymorphismCheckResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert non_image.witness is not None
    assert non_image.witness.output_row == (0, 0)

    payload["witness"]["output_row"] = [2, 0]
    with pytest.raises(ValidationError) as carrier:
        RelationalPolymorphismCheckResult.model_validate_json(
            json.dumps(payload), strict=True
        )
    assert (
        carrier.value.errors()[0]["type"]
        == "relational.polymorphism.polymorphism.witness_output"
    )

    payload["witness"]["output_row"] = [0, 0]
    payload["relation_profiles"][0]["preserved_combinations"] = 1
    with pytest.raises(ValidationError) as profile:
        RelationalPolymorphismCheckResult.model_validate_json(
            json.dumps(payload), strict=True
        )
    assert (
        profile.value.errors()[0]["type"]
        == "relational.polymorphism.polymorphism.witness_profile"
    )
