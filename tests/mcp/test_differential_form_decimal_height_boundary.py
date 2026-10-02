"""Exact form coefficient cutoffs agree across native, dispatch, and MCP."""

import asyncio
import json
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.differential_forms import (
    PolynomialDifferentialForm,
    exterior_derivative,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
)
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize("reciprocal", (False, True))
def test_decimal_interval_boundary_reaches_public_calculus(reciprocal: bool) -> None:
    operation_id = "differential_form.exterior_derivative.compute"
    coefficient = "7" + "0" * (MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS - 1)
    form: dict[str, Any] = {
        "variables": ["x"],
        "degree": "0",
        "components": [
            {
                "indices": [],
                "coefficient": {
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {
                                    "num": "-1" if reciprocal else "-" + coefficient,
                                    "den": coefficient if reciprocal else "1",
                                },
                                "exponents": [1],
                            }
                        ]
                    },
                },
            }
        ],
    }
    source = PolynomialDifferentialForm.model_validate_json(json.dumps(form))
    expected = exterior_derivative(source)
    expected_wire = expected.model_dump(mode="json")
    assert expected_wire["degree"] == "1"
    assert expected_wire["components"][0]["indices"] == [0]
    result_term = expected_wire["components"][0]["coefficient"]["polynomial"]["terms"][
        0
    ]
    assert result_term["exponents"] == [0]
    assert (
        result_term["coefficient"]
        == form["components"][0]["coefficient"]["polynomial"]["terms"][0]["coefficient"]
    )
    assert (
        invoke_operation(operation_id, {"form": form}, Catalog.open()).output
        == expected_wire
    )

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": {"form": form}}
            )
            assert not result.is_error
            assert result.structured_content is not None
            decoded = PolynomialDifferentialForm.model_validate_json(
                json.dumps(result.structured_content["output"])
            )
            assert decoded == expected

    asyncio.run(scenario())
