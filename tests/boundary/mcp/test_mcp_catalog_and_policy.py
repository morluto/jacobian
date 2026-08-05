"""MCP catalog projection and operator-policy boundary tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jacobian.adapters.mcp.server import create_server
from jacobian.capability_service import CapabilityPolicy


def test_mcp_no_retrieval_policy_is_operator_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        policy = CapabilityPolicy(profile="COMPUTE_VERIFY_NO_RETRIEVAL")
        async with Client(
            create_server(tmp_path, capability_policy=policy),
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

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            resource_result = await client.read_resource("capability://catalog")
            full_catalog = json.loads(resource_result.contents[0].text)
            all_ids = {
                descriptor["capability_id"]
                for descriptor in full_catalog["capabilities"]
            }

            listed = await client.call_tool(
                "math.find",
                {"limit": 20},
            )
            index = json.loads(listed.content[0].text)
            assert len(listed.content[0].text.encode("utf-8")) <= 16 * 1024
            assert index["catalog_digest"].startswith("sha256:")
            assert index["policy_digest"].startswith("sha256:")
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
                    {"cursor": cursor, "limit": 20},
                )
                page = json.loads(next_page.content[0].text)
                assert len(next_page.content[0].text.encode("utf-8")) <= 16 * 1024
                assert page["catalog_digest"] == index["catalog_digest"]
                indexed_ids.update(
                    descriptor["capability_id"] for descriptor in page["matches"]
                )
                cursor = page["next_cursor"]
            assert indexed_ids == all_ids

            searched = await client.call_tool(
                "math.find",
                {
                    "query": "SAT UNSAT proof",
                    "input_kind": "STRUCTURED_REQUEST",
                },
            )
            search_index = json.loads(searched.content[0].text)
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
                    "query": (
                        "finite coloring forbidden monochromatic triples exact "
                        "finite existence certified exhaustive search"
                    ),
                    "limit": 20,
                },
            )
            coloring_ids = {
                descriptor["capability_id"]
                for descriptor in json.loads(coloring_search.content[0].text)["matches"]
            }
            assert expected_sat_ids.issubset(coloring_ids)

            materialize_description = await client.call_tool(
                "math.find",
                {
                    "capability_id": "sat.cnf.materialize",
                    "view": "CONTRACT",
                },
            )
            materialize = json.loads(materialize_description.content[0].text)
            assert materialize["invocations"][0]["name"] == "finite-coloring-cnf"
            assert (
                materialize["synchronous_execution"]["remote_safe_wall_seconds_max"]
                == 150
            )
            assert {
                item["capability_id"] for item in materialize["related_capabilities"]
            }.issuperset(expected_sat_ids - {"sat.cnf.materialize"})

            first_page = await client.call_tool(
                "math.find",
                {"limit": 20},
            )
            first = json.loads(first_page.content[0].text)
            assert len(first["matches"]) <= 20
            assert first["next_cursor"] is not None
            second_page = await client.call_tool(
                "math.find",
                {"cursor": first["next_cursor"], "limit": 20},
            )
            second = json.loads(second_page.content[0].text)
            assert {
                descriptor["capability_id"] for descriptor in first["matches"]
            }.isdisjoint(
                descriptor["capability_id"] for descriptor in second["matches"]
            )

            invalid_cursor = await client.call_tool(
                "math.find",
                {
                    "query": "definitely-no-matching-capability",
                    "cursor": first["next_cursor"],
                    "limit": 20,
                },
            )
            invalid = json.loads(invalid_cursor.content[0].text)
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())
