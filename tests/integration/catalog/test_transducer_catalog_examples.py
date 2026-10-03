"""Advertised transducer example through the published catalog."""

from __future__ import annotations

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.automata.transducers.values import SubsequentialTransducer


def test_tool_is_discoverable_and_has_a_valid_example() -> None:
    operation_id = "transducer.subsequential.restrict_domain.compute"
    catalog = Catalog.open()
    tool = catalog.operation(operation_id)
    assert tool is not None
    assert len(tool.examples) == 1
    result = invoke_operation(operation_id, tool.examples[0].input, catalog)
    restricted = tool.result_type.model_validate_json(json.dumps(result.output))
    assert isinstance(restricted, SubsequentialTransducer)
    assert restricted.state_count == 2
    assert len(restricted.final_outputs) == 1
    assert restricted.final_outputs[0].state == 0
    assert restricted.final_outputs[0].output == ()
