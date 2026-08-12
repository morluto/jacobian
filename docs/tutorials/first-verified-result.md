# Compute and independently check a determinant

[Documentation home](../index.md)

This tutorial runs one exact mathematical operation and then, separately, an
operator-authorized checker. It uses only Jacobian's fixed MCP surface:
`math.find` and `math.run`.

The sequence is illustrative, not mandatory. A successful computation already
returns the determinant as a mathematical value. Run the checker only when the
investigation needs independent replay.

## Prerequisites

Use Python 3.12 and install the locked development environment from the
repository root:

```sh
uv sync --locked --dev
```

The independent determinant checker uses Python-FLINT from the development
environment.

## Run the example

Save this as `first_verified_result.py`:

```python
import asyncio
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode


STATE_DIR = Path(".jacobian-tutorial")


def rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


async def main() -> None:
    matrix = {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [rational(1), rational(0), rational(1)],
            [rational(2), rational(-1), rational(3)],
            [rational(4), rational(3), rational(2)],
        ],
    }
    server = create_server(
        STATE_DIR,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )

    async with Client(server, raise_exceptions=True) as client:
        found = await client.call_tool(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "capability_id": "matrix.determinant.compute",
                }
            },
        )
        assert isinstance(found.structured_content, dict)

        computed_call = await client.call_tool(
            "math.run",
            {
                "capability_id": "matrix.determinant.compute",
                "payload": {"matrix": matrix},
            },
        )
        assert isinstance(computed_call.structured_content, dict)
        computed = computed_call.structured_content
        assert computed["execution"]["status"] == "COMPLETED"
        assert computed["output"]["result"]["determinant"] == rational(-1)

        verified_call = await client.call_tool(
            "math.run",
            {
                "capability_id": "matrix.determinant.verify",
                "payload": {
                    "input": {"matrix": matrix},
                    "candidate": computed["output"]["result"],
                },
            },
        )
        assert isinstance(verified_call.structured_content, dict)
        verified = verified_call.structured_content
        assert verified["execution"]["status"] == "COMPLETED"
        assert verified["output"]["status"] == "VERIFIED"
        assert verified["output"]["conclusion"] == "TRUE"

        print("determinant:", computed["output"]["result"]["determinant"])
        print(
            "verification record:",
            verified["verification_record_uri"],
        )


asyncio.run(main())
```

Run it:

```sh
uv run --locked python first_verified_result.py
```

The producer computes `-1` with SymPy and returns it inline. The separate
checker independently recomputes the determinant with Python-FLINT. An accepted
check returns a verification record and resource links for its retained evidence.

## Failure states

A wrong candidate is rejected. A timeout, cancellation, malformed checker
output, unavailable provider, or interrupted checker is a non-conclusion and
cannot create a verification record. Neither an execution failure nor failure
to find evidence proves the opposite mathematical statement.

The complete typed response is in MCP `structured_content`. Text content is a
smaller human-readable projection, and resource links appear only for durable
records or evidence.

Continue with the [architecture explanation](../explanation/architecture.md)
for the ownership boundaries, or use the
[tool reference](../reference/tools.md) for the fixed MCP surface.
