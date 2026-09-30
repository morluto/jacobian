"""Koszul wire adapters unwrap into the same domain-callable package API."""

import pytest

import jacobian.math.koszul as koszul
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.koszul._tools import TOOLS
from jacobian.math.koszul.module_models import (
    ModuleKoszulHomologyMapRequest,
    ModuleKoszulMapRequest,
    ModuleKoszulSequenceLinearChangeRequest,
    ModuleKoszulTopHomologyRequest,
)


@pytest.mark.parametrize(
    "request_type",
    [
        ModuleKoszulHomologyMapRequest,
        ModuleKoszulMapRequest,
        ModuleKoszulSequenceLinearChangeRequest,
        ModuleKoszulTopHomologyRequest,
    ],
)
def test_catalog_example_matches_native_domain_call(
    request_type: type[StrictModel],
) -> None:
    tool = next(tool for tool in TOOLS if tool.request_type is request_type)
    request = request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input)
    )
    native: StrictModel
    if isinstance(request, ModuleKoszulMapRequest):
        native = koszul.module_koszul_map(
            request.algebra,
            request.source,
            request.target,
            request.sequence,
            request.map_matrix,
        )
    elif isinstance(request, ModuleKoszulHomologyMapRequest):
        native = koszul.koszul_homology_map(request.chain_map)
    elif isinstance(request, ModuleKoszulSequenceLinearChangeRequest):
        native = koszul.module_koszul_sequence_linear_change(
            request.complex, request.change_matrix
        )
    else:
        assert isinstance(request, ModuleKoszulTopHomologyRequest)
        native = koszul.module_koszul_top_homology(request.complex)
    assert tool.run(request) == native
