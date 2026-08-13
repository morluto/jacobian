"""MCP catalog projection and operator-policy boundary tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jacobian.adapters.mcp.server import create_server
from jacobian.capability_service import CapabilityPolicy
from jacobian.runtime import CheckerAuthorityMode


def test_mcp_no_retrieval_policy_is_operator_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        policy = CapabilityPolicy(profile="COMPUTE_VERIFY_NO_RETRIEVAL")
        async with Client(
            create_server(
                tmp_path,
                checker_authority=CheckerAuthorityMode.NONE,
                capability_policy=policy,
            ),
            raise_exceptions=True,
        ) as client:
            resource = await client.read_resource("capability://catalog")
            catalog = json.loads(resource.contents[0].text)
            assert catalog["policy_profile"] == "COMPUTE_VERIFY_NO_RETRIEVAL"
            assert catalog["policy_digest"] == policy.digest
            assert all(
                "retrieval" not in descriptor["tags"]
                for descriptor in catalog["capabilities"]
            )

    asyncio.run(scenario())


def test_mcp_compact_capability_index_is_searchable_and_paginated(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, checker_authority=CheckerAuthorityMode.NONE),
            raise_exceptions=True,
        ) as client:
            resource_result = await client.read_resource("capability://catalog")
            full_catalog = json.loads(resource_result.contents[0].text)
            all_ids = {
                descriptor["capability_id"]
                for descriptor in full_catalog["capabilities"]
            }
            discoverable_ids = {
                descriptor["capability_id"]
                for descriptor in full_catalog["capabilities"]
            }

            listed = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 20}},
            )
            assert isinstance(listed.structured_content, dict)
            index = listed.structured_content
            assert len(listed.content[0].text.encode("utf-8")) <= 16 * 1024
            assert len(listed.content[0].text) < len(
                json.dumps(index, separators=(",", ":"))
            )
            assert index["response_byte_limit"] == 16 * 1024
            assert len(index["matches"]) <= 20
            indexed_ids = {
                descriptor["capability_id"] for descriptor in index["matches"]
            }
            assert all(
                "input_schema" not in descriptor for descriptor in index["matches"]
            )
            cursor = index["next_cursor"]
            while cursor is not None:
                next_page = await client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "search",
                            "query": "exact",
                            "cursor": cursor,
                            "limit": 20,
                        }
                    },
                )
                assert isinstance(next_page.structured_content, dict)
                page = next_page.structured_content
                assert len(next_page.content[0].text.encode("utf-8")) <= 16 * 1024
                indexed_ids.update(
                    descriptor["capability_id"] for descriptor in page["matches"]
                )
                cursor = page["next_cursor"]
            assert len(indexed_ids) == index["total_matches"]
            assert indexed_ids <= discoverable_ids

            searched = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "SAT UNSAT proof",
                        "input_kind": "STRUCTURED_REQUEST",
                        "limit": 20,
                    }
                },
            )
            assert isinstance(searched.structured_content, dict)
            search_index = searched.structured_content
            search_ids = {
                descriptor["capability_id"] for descriptor in search_index["matches"]
            }
            expected_sat_ids = {
                "sat.cnf.materialize",
                "sat.unsat_proof.find",
                "sat.unsat_proof.verify",
            }.intersection(all_ids)
            assert expected_sat_ids.issubset(search_ids)

            coloring_search = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": (
                            "finite coloring forbidden monochromatic triples exact "
                            "finite existence certified exhaustive search"
                        ),
                        "limit": 20,
                    }
                },
            )
            assert isinstance(coloring_search.structured_content, dict)
            coloring_ids = {
                descriptor["capability_id"]
                for descriptor in coloring_search.structured_content["matches"]
            }
            assert expected_sat_ids.issubset(coloring_ids)

            materialize_description = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "capability_id": "sat.cnf.materialize",
                    }
                },
            )
            assert isinstance(materialize_description.structured_content, dict)
            materialize = materialize_description.structured_content
            assert (
                materialize["capability"]["invocation_examples"][0]["name"]
                == "finite-coloring-cnf"
            )

            first_page = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 20}},
            )
            assert isinstance(first_page.structured_content, dict)
            first = first_page.structured_content
            assert len(first["matches"]) <= 20
            assert first["next_cursor"] is not None
            second_page = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "exact",
                        "cursor": first["next_cursor"],
                        "limit": 20,
                    }
                },
            )
            assert isinstance(second_page.structured_content, dict)
            second = second_page.structured_content
            assert {
                descriptor["capability_id"] for descriptor in first["matches"]
            }.isdisjoint(
                descriptor["capability_id"] for descriptor in second["matches"]
            )

            invalid_cursor = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "definitely-no-matching-capability",
                        "cursor": first["next_cursor"],
                        "limit": 20,
                    }
                },
            )
            invalid = json.loads(invalid_cursor.content[0].text)
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())


def test_mcp_text_projection_preserves_produced_artifact_types(
    tmp_path: Path,
) -> None:
    """The agent-facing text projection must include produced_artifact_types."""

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, checker_authority=CheckerAuthorityMode.NONE),
            raise_exceptions=True,
        ) as client:
            listed = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "poset", "limit": 20}},
            )
            text = json.loads(listed.content[0].text)
            assert text["kind"] == "discovery"
            producing = [m for m in text["matches"] if "produced_artifact_types" in m]
            assert producing, (
                "text projection must preserve produced_artifact_types for at "
                "least one poset capability so agents can connect producers to "
                "consumers without structured_content"
            )

    asyncio.run(scenario())
