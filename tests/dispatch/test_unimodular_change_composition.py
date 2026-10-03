"""Public composition check for the unimodular-change operation."""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_advertised_tool_composes_after_dispatch_roundtrip() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "quadratic_form.unimodular_change.compute"
    )
    example_input = tool.examples[0].input

    # Dispatch owns the public boundary: operation-ID lookup, strict wire
    # parsing of the advertised example, and the serialized output envelope.
    dispatched = invoke_operation(tool.operation_id, example_input, Catalog.open())
    transported = tool.result_type.model_validate_json(json.dumps(dispatched.output))

    assert transported.target.diagonal_coefficients == (1, 6)
    assert transported.target.cross_terms[0].coefficient == 5
    # The retained source survives the round trip, and the returned inverse is
    # the actual unimodular change of basis (determinant +1 here), so the
    # dispatched pair still composes back to the original form.
    assert transported.source.diagonal_coefficients == (1, 2)
    assert transported.source.cross_terms[0].coefficient == 3
    assert transported.inverse.entries == ((1, -1), (0, 1))
    assert transported.matrix.entries == ((1, 1), (0, 1))
