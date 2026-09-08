"""Dispatch boundaries for exposed-face reduction of rational SDP systems."""

from __future__ import annotations

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation, parse_operation_input
from jacobian.math.matrices.semidefinite import (
    SemidefiniteFaceReduction,
    reduce_exposed_face,
)
from jacobian.math.matrices.semidefinite._models import SemidefiniteFaceReductionRequest
from jacobian.math.matrices.semidefinite._tools import TOOLS


def test_native_dispatch_and_serialized_result_parity() -> None:
    tool = TOOLS[0]
    parsed = parse_operation_input(
        SemidefiniteFaceReductionRequest, tool.examples[0].input
    )
    assert isinstance(parsed, SemidefiniteFaceReductionRequest)
    dispatched = invoke_operation(
        tool.operation_id, tool.examples[0].input, Catalog.open()
    )
    result = SemidefiniteFaceReduction.model_validate_json(
        json.dumps(dispatched.output)
    )
    assert result == reduce_exposed_face(parsed.system, parsed.multipliers)
    assert (
        SemidefiniteFaceReduction.model_validate_json(result.model_dump_json())
        == result
    )
