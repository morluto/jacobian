from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.capabilities import CapabilityDescriptor


def test_math_find_discovers_finite_expectation_by_natural_vocabulary(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            discovered = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "expectation of a discrete random variable",
                        "domain": "probability",
                        "limit": 5,
                    }
                },
            )

        assert isinstance(discovered.structured_content, dict)
        assert "probability.finite_distribution.raw_moment.compute" in {
            match["capability_id"] for match in discovered.structured_content["matches"]
        }

    asyncio.run(scenario())


def test_math_find_search_returns_compact_lexical_and_availability_facts(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "compute an exact matrix determinant",
                        "domain": "matrix",
                        "limit": 3,
                    }
                },
            )

        assert isinstance(result.structured_content, dict)
        match = next(
            item
            for item in result.structured_content["matches"]
            if item["capability_id"] == "matrix.determinant.compute"
        )
        assert match["provider_availability"] == "AVAILABLE"
        assert match["relevance_score"] > 0
        assert match["applicability"] == "NEEDS_MORE_TYPED_REQUIREMENTS"
        assert match["applicability_code"] == "FULL_REQUEST_REQUIRED"
        assert "input_schema" not in match
        assert "output_schema_summary" not in match
        assert "invocation_example" not in match
        text_match = next(
            item
            for item in json.loads(result.content[0].text)["matches"]
            if item["capability_id"] == match["capability_id"]
        )
        assert text_match["provider_availability"] == "AVAILABLE"

    asyncio.run(scenario())


def test_math_find_exact_inspection_returns_one_authoritative_descriptor(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "capability_id": "polynomial.expression.normalize",
                    }
                },
            )

        assert isinstance(result.structured_content, dict)
        structured = result.structured_content
        assert set(structured) == {
            "kind",
            "capability",
        }
        descriptor = CapabilityDescriptor.model_validate(structured["capability"])
        assert descriptor.capability_id == "polynomial.expression.normalize"
        assert descriptor.invocation_examples
        payload = descriptor.invocation_examples[0].input
        validator = Draft202012Validator(descriptor.input_schema)
        assert validator.is_valid(payload)
        assert not validator.is_valid({})
        text = json.loads(result.content[0].text)
        assert text["capability"]["capability_id"] == descriptor.capability_id
        assert "input_schema" not in text["capability"]

    asyncio.run(scenario())
