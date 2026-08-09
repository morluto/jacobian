# Discover, invoke, and verify domain capabilities

[Documentation home](../index.md)

Use this guide when you know the mathematical outcome you need but not the
installed capability ID or payload. The reliable sequence is:

1. discover the installed catalog;
2. describe the selected capability;
3. invoke it in `EXPLORE` mode;
4. inspect execution, completeness, obligations, assurance, and artifacts; and
5. when available, pass the exact result artifact to a separately described
   `VERIFY` capability.

Do not infer availability or payload fields from examples. Provider health,
configured exclusions, optional backends, and checker authorization all affect
the installed catalog.

## Discover and describe

Call `math.find` without a capability ID to receive the installed
catalog. Select by mathematical outcome and domain tags, then describe the
exact ID:

```json
{
  "capability_id": "polynomial.compute.gcd"
}
```

The descriptor is the request contract. Check its `input_schema`,
`output_schema`, modes, provider runtime, version, and fixed checker identities
before constructing a payload.

Catalog membership means available and invocable. It does not mean recommended,
release-supported, independently verified, or compatible with a different
installed version.

## Invoke a computed operation

This complete local MCP example computes a polynomial GCD and then independently
replays the result. It uses the bundled references so the verification
capability is operator-authorized and present.

```python
import asyncio
import json
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode


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
    server = create_server(
        STATE_DIR,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    async with Client(server, raise_exceptions=True) as client:
        described = await tool(
            client,
            "math.find",
            {"capability_id": "polynomial.compute.gcd"},
        )
        assert described["capability"]["modes"] == ["EXPLORE"]

        computed = await tool(
            client,
            "math.run",
            {
                "capability_id": "polynomial.compute.gcd",
                "mode": "EXPLORE",
                "payload": {
                    "left": polynomial(-1, 0, 1),
                    "right": polynomial(0, 1, 1),
                },
            },
        )
        assert computed["execution"]["status"] == "COMPLETED"
        assert computed["assurance"]["level"] == "COMPUTED"

        verification_descriptor = await tool(
            client,
            "math.find",
            {"capability_id": "polynomial.gcd.verify"},
        )
        assert verification_descriptor["capability"]["modes"] == ["VERIFY"]

        verified = await tool(
            client,
            "math.run",
            {
                "capability_id": "polynomial.gcd.verify",
                "mode": "VERIFY",
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
uv run python domain_capability.py
```

The producer's exact arithmetic and successful completion yield `COMPUTED`,
not `VERIFIED`. The second capability resolves the stored input/result lineage
and independently replays the relation. Supplying copied inline output instead
of the exact `result_uri` would lose that binding.

## Interpret bounded-search results

Bounded operations require a two-level check.

First inspect operational state:

- `COMPLETED` means the implementation returned normally;
- `TIMEOUT`, `CANCELLED`, and `ERROR` are interruptions and never conclusions.

Then inspect mathematical state:

- `completeness.status = COMPLETE` means the operation's declared completion
  predicate holds, but assurance may still be only `COMPUTED`;
- `UNKNOWN` or `PARTIAL` means the result is not complete;
- `obligations` identifies the open optimality or completeness claim; and
- the output may retain an incumbent, bounds, and a tested trace even when no
  conclusion is available.

Never use `execution.status = COMPLETED` by itself as evidence of optimality.
Keep the input, result, and obligation artifact URIs together so a later
checker or resumed investigation can address the exact open claim.

## When verification is unavailable

If describing the expected verifier returns `UNKNOWN_CAPABILITY`, do not guess
another ID or treat computed evidence as verified. Re-read the catalog. The
usual causes are disabled bundled references, an unavailable checker backend,
failed runtime measurement, configured exclusions, or a producer relation for
which no independent checker is installed.

You can still use the computed result as explicitly labeled evidence. Preserve
its provider identity, artifacts, scope, completeness, and obligations, and
report the missing verification path.

See the [domain operation library reference](../reference/domain-operation-library.md)
for the underlying producer and checker contracts.
