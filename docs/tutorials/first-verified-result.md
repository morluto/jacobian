# Find and verify a counterexample

[Documentation home](../index.md)

This tutorial uses Jacobian's public MCP surface to find and independently
verify an omitted graph path. It makes each mathematical operation visible:
materialize two artifacts, validate the claim, evaluate the candidate, find a
witness, and replay that witness with an authorized checker.

This is a teaching sequence, not a required research workflow. Agents may
compose, repeat, compare, or abandon the same capabilities in other orders.
Jacobian supplies the operations, artifacts, and trust boundary; the agent
supplies the strategy.

## Prerequisites

Use Python 3.12 and install the locked development environment from the
repository root:

```sh
uv sync --dev
```

On macOS, see the
[Z3 installation note](../../README.md#macos-and-z3) if environment setup
falls back to a source build.

## Run the capability sequence

Save the following as `first_verified_result.py`:

```python
import asyncio
import json
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode


STATE_DIR = Path(".jacobian-tutorial")


async def tool(client: Client, name: str, arguments: dict) -> dict:
    result = await client.call_tool(name, arguments)
    return json.loads(result.content[0].text)


async def main() -> None:
    server = create_server(
        STATE_DIR,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        reasoning_log_mode="off",
    )

    async with Client(server, raise_exceptions=True) as client:
        capability_ids = (
            "artifact.put",
            "claim.validate",
            "evaluate.batch",
            "witness.find",
            "witness.verify",
        )
        for capability_id in capability_ids:
            description = await tool(
                client,
                "math.find",
                {"capability_id": capability_id},
            )
            assert description["capability"]["capability_id"] == capability_id

        reference_resource = await client.read_resource("reference://catalog")
        references = json.loads(reference_resource.contents[0].text)
        graph = references["graph_paths"]

        claim = await tool(
            client,
            "math.run",
            {
                "capability_id": "artifact.put",
                "mode": "EXPLORE",
                "payload": {
                    "schema_uri": graph["claim_schema_uri"],
                    "semantics_uri": graph["semantics_uri"],
                    "payload": {
                        "claim_schema_version": "1",
                        "domain_id": "jacobian.graph-paths",
                        "domain_version": "1",
                        "semantics_uri": graph["semantics_uri"],
                        "quantifiers": [],
                        "predicate": {
                            "name": "intended_paths_complete",
                            "parameters": {"simple": True},
                        },
                        "bounds": {},
                        "required_capabilities": [
                            "Evaluator",
                            "WitnessOracle",
                        ],
                        "correspondence_status": "HUMAN_REVIEWED",
                    },
                },
            },
        )
        claim_uri = claim["output"]["artifact_uri"]

        candidate = await tool(
            client,
            "math.run",
            {
                "capability_id": "artifact.put",
                "mode": "EXPLORE",
                "payload": {
                    "schema_uri": graph["candidate_schema_uri"],
                    "semantics_uri": graph["semantics_uri"],
                    "payload": {
                        "vertices": ["s", "a", "b", "x", "t1", "t2"],
                        "arcs": [
                            ["s", "a"],
                            ["a", "x"],
                            ["s", "b"],
                            ["b", "x"],
                            ["x", "t1"],
                            ["x", "t2"],
                        ],
                        "source": "s",
                        "terminals": ["t1", "t2"],
                        "intended_paths": [
                            ["s", "a", "x", "t1"],
                            ["s", "b", "x", "t2"],
                        ],
                    },
                },
            },
        )
        candidate_uri = candidate["output"]["artifact_uri"]

        validation = await tool(
            client,
            "math.run",
            {
                "capability_id": "claim.validate",
                "mode": "EXPLORE",
                "payload": {
                    "claim_uri": claim_uri,
                    "plugin_id": graph["plugin_id"],
                },
            },
        )
        assert validation["execution"]["status"] == "COMPLETED"

        evaluation = await tool(
            client,
            "math.run",
            {
                "capability_id": "evaluate.batch",
                "mode": "EXPLORE",
                "payload": {
                    "claim_uri": claim_uri,
                    "candidate_uris": [candidate_uri],
                    "plugin_id": graph["plugin_id"],
                    "profile": "EXACT_CANDIDATE",
                    "seed": 0,
                    "wall_seconds": 30,
                },
            },
        )
        evaluated = evaluation["output"]["items"][0]["result"]
        print(
            "evaluation:",
            evaluated["conclusion"],
            evaluation["assurance"]["level"],
        )

        found = await tool(
            client,
            "math.run",
            {
                "capability_id": "witness.find",
                "mode": "EXPLORE",
                "payload": {
                    "claim_uri": claim_uri,
                    "candidate_uri": candidate_uri,
                    "plugin_id": graph["plugin_id"],
                    "witness_role": "DEFEATS_CANDIDATE",
                    "wall_seconds": 30,
                },
            },
        )
        witness_uri = found["output"]["witness_uri"]
        assert witness_uri is not None

        verified = await tool(
            client,
            "math.run",
            {
                "capability_id": "witness.verify",
                "mode": "VERIFY",
                "payload": {
                    "claim_uri": claim_uri,
                    "candidate_uri": candidate_uri,
                    "witness_uri": witness_uri,
                    "checker_id": graph["witness_checker_ids"]["graph.omitted_path"],
                },
            },
        )
        print(
            "verification:",
            verified["output"]["conclusion"],
            verified["assurance"]["level"],
        )
        print("witness:", witness_uri)


asyncio.run(main())
```

Run it:

```sh
uv run python first_verified_result.py
```

The important output is:

```text
evaluation: FALSE HEURISTIC
verification: FALSE VERIFIED
```

`evaluate.batch` found that the proposed path list is incomplete, but the
result remained heuristic evidence. `witness.find` then materialized an omitted
path. Only `witness.verify`, running in `VERIFY` mode with the authorized
`graph.omitted_path` checker, promoted the exact bound conclusion.

## Understand failure states

`TIMEOUT`, `CANCELLED`, and `ERROR` are execution states. A completed bounded
search that finds no witness is also not proof that no witness exists. Retry
with another capability or budget, change the candidate, or abandon the line
of attack; never convert operational failure or absence of evidence into a
mathematical conclusion.

The same rule applies to evaluator scores, solver statuses, generated
witnesses, and model answers. They remain evidence until an authorized
independent checker accepts artifacts bound to the exact claim, semantics,
candidate, scope, certificate format, and checker version.

## Inspect the durable state

The claim, candidate, witness, and verification record remain under
`.jacobian-tutorial/`. Their `artifact://` URIs are content-addressed, so
rerunning the script reuses identical artifacts. Each intermediate result
remains inspectable rather than disappearing inside an opaque workflow.

Continue with the [architecture explanation](../explanation/architecture.md)
to understand the trust zones, or consult the
[tool reference](../reference/tools.md) for the public capability surface.
