"""Complete streamed wheel rows retain exact public ordering at the row cap."""

import asyncio
import json
from math import gcd

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    PrimeTupleResidueWheelEnumeration,
)
from jacobian.math.number_theory.prime_affine_forms._tools import TOOLS
from jacobian.math.number_theory.prime_affine_forms.operations import residue_wheel
from jacobian.math.number_theory.prime_affine_forms.values import (
    PrimeAffineTuple,
    PrimitiveIntegerAffineForm,
)
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


def test_complete_ordered_wheel_at_public_row_boundary() -> None:
    async def scenario() -> None:
        source = PrimeAffineTuple(
            forms=(PrimitiveIntegerAffineForm(form_id="n", coefficient=1, constant=0),)
        )
        wheel = residue_wheel(source, (3, 17, 257))
        server = _build_server(state=AppState(operation_catalog=Catalog(TOOLS)))
        async with Client(server, raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "number_theory.prime_affine_forms.residue_wheel.enumerate.compute",
                    "payload": {"wheel": wheel.model_dump(mode="json")},
                },
            )
        assert not response.is_error
        assert response.structured_content is not None
        result = PrimeTupleResidueWheelEnumeration.model_validate_json(
            json.dumps(response.structured_content["output"])
        )
        assert result.wheel == wheel
        expected = tuple(r for r in range(wheel.modulus) if gcd(r, wheel.modulus) == 1)
        assert len(result.residues) == len(expected) == 8192
        assert tuple(row.residue for row in result.residues) == expected
        assert all(
            row.components == tuple(row.residue % p for p in wheel.primes)
            for row in result.residues
        )

    asyncio.run(scenario())
