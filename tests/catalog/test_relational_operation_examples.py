"""Published-example and publication checks for relational operations.

These assert that each published example is discoverable and executes with the
exact admitted result. They live here rather than under ``tests/math`` because
importing ``BUILTIN_TOOLS`` discovers and compiles every owner-local manifest,
which starts the product catalog and costs far more than the kernel work these
operations perform.
"""

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.logic.relational_structures import (
    CspSolutions,
    FiniteCspConstraint,
    FiniteCspInstance,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    RelationalPolymorphismStatus,
)
from jacobian.math.logic.relational_structures._models import RelationalProductRequest


def _tool(operation_id: str):
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)


def test_relational_polymorphism_example_round_trips():
    tool = _tool("relational.polymorphism.check")
    example = tool.examples[0]
    request = tool.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    result = tool.run(request)
    assert result.status is RelationalPolymorphismStatus.POLYMORPHISM
    assert result.polymorphism is not None
    restored = type(result).model_validate_json(result.model_dump_json(), strict=True)
    assert restored == result


def test_relational_direct_product_example_executes():
    tool = _tool("relational.structure.direct_product.compute")
    result = tool.run(RelationalProductRequest.model_validate(tool.examples[0].input))
    assert result.product.relation_tables == (((0, 3),),)


def test_csp_solution_enumeration_is_published_with_source_bound_value():
    tool = _tool("csp.solutions.enumerate.compute")
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
