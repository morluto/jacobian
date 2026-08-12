# Retrieve a Lean theorem and check a proof

[Documentation home](../index.md)

This tutorial **finds and runs ordinary Lean tools** (search/inspect
declarations), then runs the separate **checker tool** `lean.check`. Search and
inspect return declaration **values**; only the checker accepts or rejects a
proof.

## Prerequisites

Install the locked Python environment and prepare the pinned Lean runtime as
described in [Install native and formal providers](../how-to/install-native-and-formal-providers.md#lean-certificates).

## Run the public composition

Save this as `lean_declaration_discovery.py`:

```python
import asyncio
import json
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode


STATE_DIR = Path(".jacobian-lean-tutorial")


async def tool(client: Client, name: str, arguments: dict) -> dict:
    result = await client.call_tool(name, arguments)
    return json.loads(result.content[0].text)


async def main() -> None:
    server = create_server(
        STATE_DIR,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )

    async with Client(server, raise_exceptions=True) as client:
        searched = await tool(
            client,
            "math.run",
            {
                "capability_id": "lean.declaration.search",
                "payload": {
                    "environment": "MATHLIB",
                    "name_contains": "irrational_sqrt_two",
                    "result_limit": 1,
                },
            },
        )
        # Ordinary tools: primary result is declaration metadata (values).
        declaration_name = searched["output"]["result"]["declarations"][0]["name"]
        assert searched["execution"]["status"] == "COMPLETED"
        assert declaration_name

        inspected = await tool(
            client,
            "math.run",
            {
                "capability_id": "lean.declaration.inspect",
                "payload": {
                    "environment": "MATHLIB",
                    "declaration_name": declaration_name,
                },
            },
        )
        assert inspected["output"]["result"]["declaration"]["type"] == "Irrational √2"
        assert (
            inspected["output"]["result"]["environment_digest"]
            == searched["output"]["result"]["environment_digest"]
        )

        # Separate checker tool — not a mode on search/inspect.
        checked = await tool(
            client,
            "math.run",
            {
                "capability_id": "lean.check",
                "payload": {
                    "environment": "MATHLIB",
                    "statement": "Irrational (Real.sqrt 2)",
                    "proof": f"exact {declaration_name}",
                },
            },
        )
        assert checked["output"]["conclusion"] == "TRUE"
        assert checked["output"]["verification_record_uri"] is not None


asyncio.run(main())
```

Run it from the repository root:

```sh
uv run python lean_declaration_discovery.py
```

The first two calls provide premise evidence and exact environment identity.
They do not inherit the theorem's truth as a verification record. The final
call binds the proposition and proof source to the authorized Lean checker and
replays them in the pinned Mathlib environment.
