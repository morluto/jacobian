"""Dispatch and catalog boundaries for free-algebra operations.

The mathematical contracts of these operations are owned by
``tests/math/free_algebras``. Booting the complete product boundary is
forbidden there, so the published catalog and dispatch surfaces are
exercised from this owner instead.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial

SUBTRACT_OPERATION_ID = "free_algebra.polynomial.subtract.compute"
SUBSTITUTE_OPERATION_ID = "free_algebra.polynomial.substitute.compute"
WORD_OPERATION_IDS = (
    "free_word.concatenate.compute",
    "free_word.power.compute",
    "free_word.reverse.compute",
    "free_word.prefixes.compute",
    "free_word.suffixes.compute",
    "free_word.factors.compute",
    "free_word.overlaps.compute",
    "free_word.order.compare",
    "free_word.substitute.compute",
)


def test_subtract_catalog_example_executes_and_round_trips_canonical_result() -> None:
    catalog = Catalog.open()
    tool = catalog.operation(SUBTRACT_OPERATION_ID)
    assert tool is not None and tool.examples
    response = invoke_operation(SUBTRACT_OPERATION_ID, tool.examples[0].input, catalog)
    result = tool.result_type.model_validate_json(json.dumps(response.output))
    assert isinstance(result, FreeAlgebraPolynomial)
    assert {term.word: term.coefficient.as_fraction() for term in result.terms} == {
        ("x",): Fraction(1),
        ("x", "y"): Fraction(3, 2),
        ("y",): Fraction(-1),
    }


def test_substitute_catalog_example_executes_through_dispatch() -> None:
    catalog = Catalog.open()
    tool = catalog.operation(SUBSTITUTE_OPERATION_ID)
    assert tool is not None and tool.examples
    public = invoke_operation(SUBSTITUTE_OPERATION_ID, tool.examples[0].input, catalog)
    assert public.output == {
        "alphabet": ["a", "b"],
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "word": []}],
    }


def test_word_tool_examples_execute_through_catalog() -> None:
    catalog = Catalog.open()
    for operation_id in WORD_OPERATION_IDS:
        operation = catalog.operation(operation_id)
        assert operation is not None and operation.examples
        for example in operation.examples:
            result = invoke_operation(operation_id, example.input, catalog)
            validated = operation.result_type.model_validate(result.output)
            assert validated.model_dump(mode="json") == result.output
