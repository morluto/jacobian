"""Nested simplicial face guidance is visible before strict MCP consumption."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator
from mcp.types import TextContent

from jacobian.math.topology._homology import SimplicialHomologyResult
from jacobian.math.topology._models import (
    ChainComplexResult,
    FacesInDimension,
    FiniteSimplicialComplex,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _definition(schema: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    definition = schema["$defs"][reference["$ref"].removeprefix("#/$defs/")]
    assert isinstance(definition, dict)
    return definition


def test_nested_face_guidance_and_canonical_example_recover_through_mcp() -> None:
    async def scenario() -> None:
        homology_id = "topology.simplicial_homology.compute"
        chain_id = "topology.simplicial_complex.chain_complex.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            example: dict[str, Any] = {}
            for operation_id in (
                homology_id,
                chain_id,
                "cellular_sheaf.cohomology.compute",
            ):
                found = await client.call_tool(
                    "math.find", {"operation_id": operation_id}
                )
                assert not found.is_error
                schema = found.structured_content["operation"]["input_schema"]
                parent = schema
                # The sheaf consumer adds another nested canonical value before
                # FiniteSimplicialComplex -> faces_by_dimension[] -> FacesInDimension.
                if operation_id == "cellular_sheaf.cohomology.compute":
                    parent = _definition(schema, parent["properties"]["sheaf"])
                complex_schema = _definition(schema, parent["properties"]["complex"])
                faces_schema = _definition(
                    schema, complex_schema["properties"]["faces_by_dimension"]["items"]
                )
                assert "lexicographic" in faces_schema["description"]
                description = faces_schema["properties"]["faces"]["description"]
                assert "exactly dimension + 1 vertices" in description
                assert "sorted lexicographic vertex-label order" in description
                assert "Faces must be unique" in description
                assert "outer list must be lexicographically ordered" in description
                assert (
                    "dimension + 1"
                    in faces_schema["properties"]["dimension"]["description"]
                )
                example = faces_schema["examples"][0]
                Draft202012Validator(faces_schema).validate(example)
                parsed = FacesInDimension.model_validate_json(
                    json.dumps(example), strict=True
                )
                assert parsed.dimension == 1
                assert parsed.faces == (("a", "b"), ("a", "c"), ("b", "c"))

            canonicalized = await client.call_tool(
                "math.run",
                {
                    "operation_id": "topology.simplicial_complex.canonicalize",
                    "payload": {
                        "vertices": ["c", "a", "b"],
                        "facets": [["c", "b"], ["b", "a"], ["c", "a"]],
                    },
                },
            )
            assert not canonicalized.is_error
            canonical = canonicalized.structured_content["output"]["complex"]
            decoded = FiniteSimplicialComplex.model_validate_json(
                json.dumps(canonical), strict=True
            )
            assert decoded.f_vector == (3, 3)
            # The emitted nested example is directly usable in this canonical value.
            assert canonical["faces_by_dimension"][1] == example

            malformed = deepcopy(canonical)
            malformed["faces_by_dimension"][1]["faces"].reverse()
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": homology_id,
                    "payload": {"complex": malformed, "prime": 2},
                },
            )
            assert rejected.is_error
            content = rejected.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "INVALID_REQUEST"
            assert diagnostic["errors"] == [
                {
                    "code": "topology.require_canonical_faces_2",
                    "location": ["complex", "faces_by_dimension", 1],
                    "message": "faces must be unique and lexicographically ordered",
                }
            ]

            # Recover using exactly the face group emitted by nested inspection.
            malformed["faces_by_dimension"][1] = example
            assert malformed == canonical
            homology = await client.call_tool(
                "math.run",
                {
                    "operation_id": homology_id,
                    "payload": {"complex": malformed, "prime": 2},
                },
            )
            assert not homology.is_error
            result = SimplicialHomologyResult.model_validate_json(
                json.dumps(homology.structured_content["output"]), strict=True
            )
            assert result.complex == decoded
            assert tuple(group.betti_number for group in result.groups) == (1, 1)

            # Feed the same serialized canonical value unchanged to another owner consumer.
            chain = await client.call_tool(
                "math.run",
                {"operation_id": chain_id, "payload": {"complex": canonical}},
            )
            assert not chain.is_error
            chain_result = ChainComplexResult.model_validate_json(
                json.dumps(chain.structured_content["output"]), strict=True
            )
            assert chain_result.complex == decoded
            assert chain_result.canonical_value.basis_sizes == (3, 3)
            # Edges (a,b), (a,c), (b,c), oriented by their sorted labels.
            assert chain_result.canonical_value.differential_matrices == (
                ((-1, -1, 0), (1, 0, -1), (0, 1, 1)),
            )

    asyncio.run(scenario())
