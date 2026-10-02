"""Composite characteristics retain actionable MCP diagnostics and recovery."""

import asyncio
import json

import pytest
from mcp.types import TextContent

from jacobian.math.topology.chain_complexes.values import (
    HomologyResult,
    VerificationResult,
)
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize(
    "operation_id",
    ["chain_complex.homology.compute", "chain_complex.verify_differential.compute"],
)
def test_composite_characteristic_diagnostic_and_same_session_recovery(
    operation_id: str,
) -> None:
    async def scenario() -> None:
        source = {
            "coefficient_ring": "GF_p",
            "prime": 4,
            "degree_min": 0,
            "degree_max": 0,
            "basis_sizes": [1],
            "differential_matrices": [],
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            found = await client.call_tool("math.find", {"operation_id": operation_id})
            assert not found.is_error
            rejected = await client.call_tool(
                "math.run",
                {"operation_id": operation_id, "payload": {"complex": source}},
            )
            assert rejected.is_error
            assert rejected.structured_content is None
            assert isinstance(rejected.content[0], TextContent)
            assert '"INVALID_REQUEST"' in rejected.content[0].text
            diagnostic = json.loads(
                rejected.content[0].text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["stage"] == "operation_validation"
            assert diagnostic["errors"] == [
                {
                    "code": "chain_complex.prime_not_prime",
                    "location": ["complex", "prime"],
                    "message": "prime 4 is not prime",
                }
            ]
            for prime in (2, 3):
                source["prime"] = prime
                result = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {"complex": source}},
                )
                assert not result.is_error
                assert result.structured_content is not None
                output = result.structured_content["output"]
                assert output["complex"] == source
                restored: HomologyResult | VerificationResult
                if operation_id == "chain_complex.homology.compute":
                    assert output["homology_groups"] == [
                        {
                            "kind": "FIELD_VECTOR_SPACE",
                            "degree": 0,
                            "cycle_rank": 1,
                            "boundary_rank": 0,
                            "betti_number": 1,
                        }
                    ]
                    restored = HomologyResult.model_validate_json(
                        json.dumps(output), strict=True
                    )
                else:
                    assert output["is_valid"] is True
                    restored = VerificationResult.model_validate_json(
                        json.dumps(output), strict=True
                    )
                assert restored.model_dump(mode="json") == output

    asyncio.run(scenario())


def test_simplicial_rp2_chain_output_composes_over_integers_and_prime_fields() -> None:
    async def scenario() -> None:
        facets = [
            ["0", "1", "2"],
            ["0", "1", "3"],
            ["0", "2", "4"],
            ["0", "3", "5"],
            ["0", "4", "5"],
            ["1", "2", "5"],
            ["1", "3", "4"],
            ["2", "3", "4"],
            ["2", "3", "5"],
            ["1", "4", "5"],
        ]
        async with Client(create_server(), raise_exceptions=False) as client:
            canonical = await client.call_tool(
                "math.run",
                {
                    "operation_id": "topology.simplicial_complex.canonicalize",
                    "payload": {"vertices": list("012345"), "facets": facets},
                },
            )
            assert not canonical.is_error
            assert canonical.structured_content is not None
            complex_value = canonical.structured_content["output"]["complex"]
            assert complex_value["f_vector"] == [6, 15, 10]
            for prime in (None, 2, 3):
                chain = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "topology.simplicial_complex.chain_complex.compute",
                        "payload": {
                            "complex": json.loads(json.dumps(complex_value)),
                            "coefficient_ring": "INTEGER"
                            if prime is None
                            else "PRIME_FIELD",
                            "prime": prime,
                        },
                    },
                )
                assert not chain.is_error
                assert chain.structured_content is not None
                chain_output = chain.structured_content["output"]
                assert chain_output["complex"] == complex_value
                source = json.loads(json.dumps(chain_output["canonical_value"]))
                homology = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "chain_complex.homology.compute",
                        "payload": {"complex": source},
                    },
                )
                assert not homology.is_error
                assert homology.structured_content is not None
                output = homology.structured_content["output"]
                assert output["complex"] == source
                restored = HomologyResult.model_validate_json(
                    json.dumps(output), strict=True
                )
                assert restored.model_dump(mode="json") == output
                groups = output["homology_groups"]
                if prime is None:
                    assert [group["free_rank"] for group in groups] == [1, 0, 0]
                    assert [group["torsion_invariant_factors"] for group in groups] == [
                        [],
                        ["2"],
                        [],
                    ]
                    generator = groups[1]["torsion_generators"][0]
                    cycle = [int(v) for v in generator["cycle"]["coefficients"]]
                    bounding = [
                        int(v) for v in generator["bounding_chain"]["coefficients"]
                    ]
                    outgoing, incoming = source["differential_matrices"]
                    assert all(
                        sum(int(v) * c for v, c in zip(row, cycle, strict=True)) == 0
                        for row in outgoing
                    )
                    assert [
                        sum(int(v) * b for v, b in zip(row, bounding, strict=True))
                        for row in incoming
                    ] == [2 * c for c in cycle]
                else:
                    assert [group["betti_number"] for group in groups] == (
                        [1, 1, 1] if prime == 2 else [1, 0, 0]
                    )

    asyncio.run(scenario())
