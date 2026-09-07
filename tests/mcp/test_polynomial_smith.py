"""QQ[t] Smith native/MCP parity and resource-rejection recovery."""

import asyncio
import json

from jacobian.math.matrices.certified_snf._models import PolynomialSmithRequest
from jacobian.math.matrices.certified_snf.polynomial import (
    polynomial_smith_decomposition,
)
from jacobian.math.matrices.symbolic import RationalPolynomialMatrix
from jacobian.mcp.server import create_server
from mcp import Client


def test_rectangular_torsion_native_mcp_parity_and_recovery() -> None:
    async def scenario() -> None:
        def entry(degree: int | None) -> dict[str, object]:
            return {
                "variables": ["t"],
                "polynomial": {
                    "terms": []
                    if degree is None
                    else [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [degree],
                        },
                    ]
                },
            }

        payload = {
            "matrix": {
                "variables": ["t"],
                "row_count": 2,
                "column_count": 3,
                "entries": [[entry(1), entry(2), entry(None)], [entry(None)] * 3],
            }
        }
        native = polynomial_smith_decomposition(
            PolynomialSmithRequest.model_validate_json(json.dumps(payload)).matrix
        )
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = RationalPolynomialMatrix(
                variables=("t",), row_count=0, column_count=129, entries=()
            )
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.normal_form.smith.polynomial.compute",
                    "payload": {"matrix": invalid.model_dump(mode="json")},
                },
            )
            assert rejected.is_error
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.normal_form.smith.polynomial.compute",
                    "payload": payload,
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == native.model_dump(mode="json")

    asyncio.run(scenario())
