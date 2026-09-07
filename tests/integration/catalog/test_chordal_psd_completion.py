"""Dispatch, canonical output and schema parity for exact completion."""

import json

from jsonschema import validate

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.completion import complete_chordal_psd
from jacobian.math.matrices.completion._models import (
    ChordalPSDCompletionRequest,
    ChordalPSDCompletionResult,
)
from jacobian.math.matrices.completion._tools import TOOLS


def test_public_completion_native_parity_and_matrix_composition() -> None:
    payload = TOOLS[0].examples[0].input
    validate(payload, ChordalPSDCompletionRequest.model_json_schema())
    source = ChordalPSDCompletionRequest.model_validate_json(json.dumps(payload)).matrix
    native = complete_chordal_psd(source).model_dump(mode="json")
    catalog = Catalog.open()
    wire = invoke_operation("matrix.chordal_psd_completion.compute", payload, catalog)
    assert wire.output == native
    validate(wire.output, ChordalPSDCompletionResult.model_json_schema())
    inertia = invoke_operation(
        "matrix.inertia.compute", {"matrix": wire.output["completion"]}, catalog
    )
    assert inertia.output["n_positive"] == 3
    assert inertia.output["n_negative"] == 0
