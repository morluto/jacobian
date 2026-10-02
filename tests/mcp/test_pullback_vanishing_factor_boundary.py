"""Zero pullbacks and genuine output growth agree across public boundaries."""

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.differential_forms import (
    PolynomialDifferentialForm,
    PolynomialMap,
    pullback,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _polynomial(axis: list[str], exponents: list[int]) -> dict[str, Any]:
    return {
        "variables": axis,
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": exponents}]
        },
    }


@pytest.mark.parametrize("same_dimension", (False, True))
def test_zero_factor_avoids_large_public_substitution(same_dimension: bool) -> None:
    axis = ["s", "t"] if same_dimension else ["s"]
    image = _polynomial(axis, [2, 0] if same_dimension else [2])
    mapping = {
        "source_variables": axis,
        "target_variables": ["x", "y"],
        "images": [image, image],
    }
    form = {
        "variables": ["x", "y"],
        "degree": "2",
        "components": [
            {"indices": [0, 1], "coefficient": _polynomial(["x", "y"], [256, 0])}
        ],
    }
    native = pullback(
        PolynomialMap.model_validate_json(json.dumps(mapping)),
        PolynomialDifferentialForm.model_validate_json(json.dumps(form)),
    )
    expected = native.model_dump(mode="json")
    assert expected == {"variables": axis, "degree": "2", "components": []}
    operation = "differential_form.pullback.compute"
    payload = {"mapping": mapping, "form": form}
    assert invoke_operation(operation, payload, Catalog.open()).output == expected

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            decoded = PolynomialDifferentialForm.model_validate_json(
                json.dumps(result.structured_content["output"])
            )
            assert decoded == native
            nonzero = {
                "mapping": {
                    "source_variables": ["s", "t"],
                    "target_variables": ["x", "y"],
                    "images": [
                        _polynomial(["s", "t"], [2, 0]),
                        _polynomial(["s", "t"], [0, 1]),
                    ],
                },
                "form": {
                    **form,
                    "components": [
                        {
                            "indices": [0, 1],
                            "coefficient": _polynomial(["x", "y"], [128, 0]),
                        }
                    ],
                },
            }
            with pytest.raises(OperationResourceAdmissionError) as native_error:
                invoke_operation(operation, nonzero, Catalog.open())
            refused = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": nonzero}
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                diagnostic["errors"][0]["code"]
                == native_error.value.errors()[0]["type"]
            )

    asyncio.run(scenario())
