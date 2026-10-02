"""Shared set-function discovery exposes the coupled complete-table contract."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.optimization.submodular._models import SetFunction


@pytest.mark.parametrize("suffix", ["evaluate", "monotonicity", "submodularity"])
def test_set_function_schemas_explain_complete_coverage(suffix: str) -> None:
    operation = Catalog.open().inspect(f"combinatorics.set_function.{suffix}")
    assert operation is not None
    definitions = operation.input_schema["$defs"]
    function = definitions["SetFunction"]
    entry = definitions["SetFunctionEntry"]
    ground = function["properties"]["ground_set_size"]
    entries = function["properties"]["entries"]
    assert "2^n" in ground["description"]
    assert "n=0" in ground["description"]
    assert "2^ground_set_size" in entries["description"]
    assert "exactly once" in entries["description"]
    assert "element order" in entries["description"]
    assert "no repeated elements" in entry["properties"]["subset"]["description"]
    assert "ground_set_size - 1" in entry["properties"]["subset"]["description"]
    assert entries["maxItems"] == 2 ** ground["maximum"]
    example = {"ground_set_size": 1, "entries": entries["examples"][0]}
    parsed = SetFunction.model_validate_json(json.dumps(example))
    assert tuple(row.subset for row in parsed.entries) == ((), (0,))
