"""Independent finite oracle for source-bound CSP solution enumeration."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.relational_structures import (
    CspSolutions,
    FiniteCspConstraint,
    FiniteCspInstance,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    enumerate_csp_solutions,
)


def _direct_solutions(instance: FiniteCspInstance) -> tuple[tuple[int, ...], ...]:
    """Evaluate named constraints directly, without using relational conversion."""

    relation_by_id = {
        symbol.symbol_id: set(table)
        for symbol, table in zip(
            instance.template.signature, instance.template.relation_tables, strict=True
        )
    }
    return tuple(
        assignment
        for assignment in product(
            range(instance.template.carrier_size), repeat=instance.variable_count
        )
        if all(
            tuple(assignment[variable] for variable in constraint.scope)
            in relation_by_id[constraint.symbol_id]
            for constraint in instance.constraints
        )
    )


def test_operation_is_published_with_source_bound_solution_value() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "csp.solutions.enumerate.compute"
    )
    assert tool.request_type is FiniteCspInstance
    assert tool.result_type is CspSolutions
    instance = FiniteCspInstance(
        template=FiniteRelationalStructure(
            carrier_size=2,
            signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
            relation_tables=(((0, 1), (1, 0)),),
        ),
        variable_count=2,
        constraints=(
            FiniteCspConstraint(constraint_id="edge", symbol_id="E", scope=(0, 1)),
        ),
    )
    assert tool.run(instance).assignments == ((0, 1), (1, 0))


@pytest.mark.parametrize("carrier_relation_mask", range(16))
def test_small_binary_templates_match_direct_assignment_oracle(
    carrier_relation_mask: int,
) -> None:
    carrier_pairs = tuple(product(range(2), repeat=2))
    relation = tuple(
        pair
        for bit, pair in enumerate(carrier_pairs)
        if carrier_relation_mask & (1 << bit)
    )
    template = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(relation,),
    )
    for variable_count in range(3):
        scopes = tuple(product(range(variable_count), repeat=2))
        # Include every singleton scope and every ordered pair of occurrences;
        # occurrence IDs stay distinct even when scopes repeat.
        occurrences: list[tuple[tuple[int, ...], ...]] = [()]
        occurrences.extend((scope,) for scope in scopes)
        occurrences.extend(product(scopes, repeat=2))
        for selected_scopes in occurrences:
            instance = FiniteCspInstance(
                template=template,
                variable_count=variable_count,
                constraints=tuple(
                    FiniteCspConstraint(
                        constraint_id=f"c{index}", symbol_id="E", scope=scope
                    )
                    for index, scope in enumerate(selected_scopes)
                ),
            )
            result = enumerate_csp_solutions(instance)
            assert result.instance == instance
            assert result.assignments == _direct_solutions(instance)
            assert result.total_candidates == 2**variable_count


def test_empty_axes_nullary_relations_and_named_duplicate_occurrences() -> None:
    true_template = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="T", arity=0),),
        relation_tables=(((),),),
    )
    empty_instance = FiniteCspInstance(
        template=true_template,
        variable_count=0,
        constraints=(
            FiniteCspConstraint(constraint_id="truth", symbol_id="T", scope=()),
        ),
    )
    result = enumerate_csp_solutions(empty_instance)
    assert result.assignments == ((),)
    assert result.total_candidates == 1

    duplicate_instance = FiniteCspInstance(
        template=FiniteRelationalStructure(
            carrier_size=2,
            signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
            relation_tables=(((0, 0), (1, 1)),),
        ),
        variable_count=1,
        constraints=(
            FiniteCspConstraint(constraint_id="first", symbol_id="E", scope=(0, 0)),
            FiniteCspConstraint(constraint_id="second", symbol_id="E", scope=(0, 0)),
        ),
    )
    duplicate_result = enumerate_csp_solutions(duplicate_instance)
    assert tuple(
        item.constraint_id for item in duplicate_result.instance.constraints
    ) == (
        "first",
        "second",
    )
    assert duplicate_result.assignments == ((0,), (1,))

    empty_target_instance = FiniteCspInstance(
        template=FiniteRelationalStructure(carrier_size=0),
        variable_count=1,
        constraints=(),
    )
    assert enumerate_csp_solutions(empty_target_instance).assignments == ()

    false_template = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="F", arity=0),),
        relation_tables=((),),
    )
    false_instance = FiniteCspInstance(
        template=false_template,
        variable_count=0,
        constraints=(
            FiniteCspConstraint(constraint_id="false", symbol_id="F", scope=()),
        ),
    )
    assert enumerate_csp_solutions(false_instance).assignments == ()


def test_rejects_search_space_before_enumerating_assignments() -> None:
    instance = FiniteCspInstance(
        template=FiniteRelationalStructure(carrier_size=3),
        variable_count=11,
        constraints=(),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        enumerate_csp_solutions(instance)
    assert error.value.errors()[0]["type"] == "relational.homomorphism.search_space"


def test_result_structure_rejects_malformed_assignment_payload() -> None:
    instance = FiniteCspInstance(
        template=FiniteRelationalStructure(carrier_size=2),
        variable_count=1,
        constraints=(),
    )
    payload = enumerate_csp_solutions(instance).model_dump()
    payload["assignments"] = [[1], [0]]
    with pytest.raises(ValueError, match="lexicographically ordered"):
        CspSolutions.model_validate(payload)
    payload["assignments"] = [[2], [3]]
    with pytest.raises(ValueError, match="template-valued"):
        CspSolutions.model_validate(payload)
