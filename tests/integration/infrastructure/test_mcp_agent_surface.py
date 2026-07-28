from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from mcp import Client

from jacobian.adapters.mcp.server import create_server

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2] / "snapshots" / "mcp_agent_surface.sha256"
)
FIXTURE_ENTRYPOINT = "tests.fixtures.capability_functions:create_adapter"


def _dump_models(models: list[Any], *, key: str) -> list[dict[str, Any]]:
    return sorted(
        (model.model_dump(mode="json") for model in models),
        key=lambda item: item[key],
    )


def _normalize_discovery(payload: dict[str, Any]) -> dict[str, Any]:
    digest = payload.get("catalog_digest")
    assert isinstance(digest, str) and digest.startswith("sha256:")
    return {**payload, "catalog_digest": "sha256:<catalog>"}


async def _capture_surface(state_dir: Path) -> dict[str, Any]:
    server = create_server(
        state_dir,
        capability_adapter_entrypoints=(FIXTURE_ENTRYPOINT,),
    )
    async with Client(server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        instructions = await client.read_resource("jacobian://instructions")
        discovered = await client.call_tool(
            "capability.describe",
            {
                "query": "increment an integer",
                "domain": "fixture",
                "mode": "EXPLORE",
                "limit": 5,
            },
        )
        browsed = await client.call_tool(
            "capability.describe",
            {"domain": "fixture", "limit": 5},
        )
        exact = await client.call_tool(
            "capability.describe",
            {"capability_id": "fixture.increment"},
        )
        exact_payload = json.loads(exact.content[0].text)
        return {
            "server": {
                "name": server.name,
                "title": server.title,
                "description": server.description,
                "instructions": server.instructions,
                "version": server.version,
            },
            "tools": _dump_models(tools.tools, key="name"),
            "resources": _dump_models(resources.resources, key="uri"),
            "resource_templates": _dump_models(
                templates.resource_templates,
                key="uri_template",
            ),
            "prompts": _dump_models(prompts.prompts, key="name"),
            "operating_resource": {
                "uri": "jacobian://instructions",
                "mime_type": instructions.contents[0].mime_type,
                "text": instructions.contents[0].text,
            },
            "representative_discovery": {
                "query": _normalize_discovery(json.loads(discovered.content[0].text)),
                "browse": _normalize_discovery(json.loads(browsed.content[0].text)),
                "exact": {
                    "kind": exact_payload["kind"],
                    "view": exact_payload["view"],
                    "capability_id": exact_payload["capability"]["capability_id"],
                    "input_schema_summary": exact_payload["capability"][
                        "input_schema_summary"
                    ],
                    "output_schema_summary": exact_payload["capability"][
                        "output_schema_summary"
                    ],
                    "has_invocation_examples": exact_payload["capability"][
                        "has_invocation_examples"
                    ],
                    "scope_rule": exact_payload["scope_rule"],
                    "next_views": exact_payload["next_views"],
                },
            },
        }


def test_complete_agent_facing_mcp_surface_matches_snapshot(tmp_path: Path) -> None:
    actual = asyncio.run(_capture_surface(tmp_path))
    canonical = json.dumps(
        actual,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    actual_digest = hashlib.sha256(canonical).hexdigest()
    expected_digest = SNAPSHOT_PATH.read_text(encoding="utf-8").strip()

    assert actual_digest == expected_digest
