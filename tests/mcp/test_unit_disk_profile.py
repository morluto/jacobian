"""Boundary-sensitive profile through the live MCP projection."""

import asyncio

from jacobian.math.polynomials.unit_circle import UnitDiskProfile, unit_disk_profile
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.mcp.server import create_server
from mcp import Client


def test_exact_profile_native_mcp_parity_and_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            source = RationalPolynomial.model_validate(
                {
                    "variables": ["z"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": c, "den": 1}, "exponents": [i]}
                            for i, c in [(4, 1), (3, 1), (2, -1), (1, 1), (0, 1)]
                        ]
                    },
                }
            )
            payload = {"polynomial": source.model_dump(mode="json")}
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.root_location.unit_disk_profile.compute",
                    "payload": {
                        "polynomial": {"variables": ["z"], "polynomial": {"terms": []}}
                    },
                },
            )
            assert rejected.is_error
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.root_location.unit_disk_profile.compute",
                    "payload": payload,
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == unit_disk_profile(
                source
            ).model_dump(mode="json")
            import json

            value = UnitDiskProfile.model_validate_json(
                json.dumps(result.structured_content["output"])
            )
            assert (value.inside, value.on, value.outside) == (1, 2, 1)
            assert unit_disk_profile(value.polynomial) == value

    asyncio.run(scenario())
