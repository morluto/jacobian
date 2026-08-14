# Discover, invoke, and check domain math tools

[Documentation home](../index.md)

Use this guide when you know the mathematical outcome you need but not the
built-in operation ID or payload. The product surface is
[search and execute](../explanation/architecture.md#search-and-execute):

1. **Search or inspect** with `math.find`;
2. **Execute** the ordinary tool with `math.run` and read the **mathematical
   value** in `output`;
3. use execution status (and optional artifacts) for failures and handoffs; and
4. when independent checking matters, run a **separate checker tool** (often a
   `*.verify` ID)—not a second mode on the producer.

Do not infer availability or payload fields from examples. The active catalog
is authoritative; optional external executables and checker authorization may
affect which exceptional operations it contains.

## Discover and describe

Call `math.find` with a search request to receive catalog matches.
Select by mathematical outcome and domain tags, then inspect the exact ID:

```json
{
  "request": {
    "op": "inspect",
    "operation_id": "polynomial.compute.gcd"
  }
}
```

The descriptor is the request contract. Check its `input_schema`,
`output_schema`, version, accepted input kinds, artifact types, and examples
before constructing a payload. Runtime and checker executable identities are
operator-owned internal state and are not exposed in operation descriptors.

Catalog membership means available and invocable. It does not mean recommended,
release-supported, independently verified, or compatible with a different
installed version.

## Invoke a producer tool, then a checker tool

This complete local MCP example runs two **different** math tools: a GCD
producer, then an independent GCD checker. The important split is the tool ID
and the checker verdict, not a mode on the producer.

It uses the bundled references so the checker tool is operator-authorized and
present.

```python
import asyncio
import json
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server
from jacobian.operator_lifecycle import initialize_state


STATE_DIR = Path(".jacobian-domain-how-to")


def q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def polynomial(*coefficients_ascending: int) -> dict:
    return {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": q(coefficient),
                    "exponents": [exponent],
                }
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


async def tool(client: Client, name: str, arguments: dict) -> dict:
    result = await client.call_tool(name, arguments)
    return json.loads(result.content[0].text)


async def main() -> None:
    initialize_state(STATE_DIR)
    server = create_server(STATE_DIR)
    async with Client(server, raise_exceptions=True) as client:
        # Ordinary producer tool.
        described = await tool(
            client,
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "operation_id": "polynomial.compute.gcd",
                }
            },
        )
        assert described["operation"]["operation_id"] == "polynomial.compute.gcd"

        computed = await tool(
            client,
            "math.run",
            {
                "operation_id": "polynomial.compute.gcd",
                "payload": {
                    "left": polynomial(-1, 0, 1),
                    "right": polynomial(0, 1, 1),
                },
            },
        )
        assert computed["execution"]["status"] == "COMPLETED"

        # Separate checker tool.
        verification_descriptor = await tool(
            client,
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "operation_id": "polynomial.gcd.verify",
                }
            },
        )
        assert (
            verification_descriptor["operation"]["operation_id"]
            == "polynomial.gcd.verify"
        )

        verified = await tool(
            client,
            "math.run",
            {
                "operation_id": "polynomial.gcd.verify",
                "payload": {
                    "input": {
                        "left": polynomial(-1, 0, 1),
                        "right": polynomial(0, 1, 1),
                    },
                    "candidate": computed["output"]["result"],
                },
            },
        )
        assert verified["output"]["status"] == "VERIFIED"
        assert verified["output"]["verification_record_uri"] is not None


asyncio.run(main())
```

Run it from the repository root:

```sh
uv run python domain_operation.py
```

The producer's exact arithmetic and successful completion yield `COMPUTED`,
not `VERIFIED`. The second operation independently replays the exact typed
input/candidate pair and binds their canonical digests.

## Interpret bounded-search results

Bounded operations require a two-level check.

First inspect operational state:

- `COMPLETED` means the implementation returned normally;
- `TIMEOUT`, `CANCELLED`, and `ERROR` are interruptions and never conclusions.

Then inspect mathematical state:

- the operation's typed output owns its completion or coverage status;
- `UNKNOWN`, `INCOMPLETE`, or a domain-specific partial status is not a
  negative conclusion; and
- the output may retain an incumbent, bounds, and a tested trace even when no
  conclusion is available.

Never use `execution.status = COMPLETED` by itself as evidence of optimality.
Keep the input and result artifact URIs together so a later checker can address
the exact typed subject and candidate. The generic response envelope does not
create an obligation lifecycle.

## When verification is unavailable

If describing the expected verifier returns `UNKNOWN_OPERATION`, do not guess
another ID or treat computed evidence as verified. Re-read the catalog. The
usual causes are an unavailable optional checker executable, omitted checker
authorization, visibility policy, or a producer relation for which Jacobian
ships no independent checker.

You can still use the computed result as explicitly labeled evidence. Preserve
its typed value and artifacts, and report the missing checker operation.

See the [domain operation library reference](../reference/domain-operation-library.md)
for the underlying producer and checker contracts.
