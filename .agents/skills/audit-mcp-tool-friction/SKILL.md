---
name: audit-mcp-tool-friction
description: Audit existing MCP tools for execution friction after an agent has reached the tool. Use when reviewing public JSON Schemas, Pydantic validation, operation examples, error recovery, completed agent traces, or benchmark runs to explain malformed calls, retries, or unusable results. Do not use to measure whether agents discover or select a tool; use evaluate-mcp-tool-adoption for that.
---

# Audit MCP Tool Friction

Audit whether an agent that has chosen a tool can form a valid request, recover
from a failure, and use the returned value. This is a public-contract audit,
not a call-count target and not a test of model willingness to use a tool.

## Set the boundary

Record the tool, revision, client, task or trace, and public inputs inspected.
Keep these questions separate:

- **Adoption:** whether the agent saw, discovered, or selected a tool.
- **Friction:** whether the selected tool made its contract, valid input, error
  recovery, and result use clear enough to execute correctly.
- **Capability:** whether an operation actually covers the requested work.

Do not call a successful Python fallback evidence that a tool contract is easy
to use. Do not call a single malformed request a product defect without
checking what the public contract exposed beforehand.

## Inspect the public contract

For each relevant operation, inspect the generated MCP input and output
schemas, tool description, discovery/inspection result, operation examples,
and error messages. Trace every non-obvious requirement to its owner:

| Contract element | Check |
| --- | --- |
| Input expression | Required values, units, encodings, ordering, and bounds are visible before the call. |
| Schema projection | Constraints representable in JSON Schema appear there with useful field descriptions and examples. |
| Semantic validation | Rules enforced only by a model validator have an explicit description and minimal valid example. |
| Recovery | Errors identify the field and state the smallest concrete correction. |
| Result use | Result shape, exactness, limits, and reconstruction information let the caller continue without guessing. |

Pydantic does not infer JSON-Schema descriptions or examples from custom model
validators. Add `Field(description=..., examples=...)` or appropriate
`json_schema_extra` when a semantic invariant cannot be encoded as a standard
JSON-Schema constraint. Keep canonical output rules when they support stable
composition; improve the input guidance before weakening the invariant.

## Read a trace as a recovery journey

Capture the exact operation ID, first payload, typed result or error, retry,
and final use of the value. Classify the first failure narrowly:

- **Hidden representation rule:** required ordering, encoding, or cross-field
  relation was not visible in the public contract.
- **Insufficient example or description:** the rule was technically present
  but not concrete enough for an ordinary first request.
- **Unhelpful recovery:** the error lacks a field, cause, or actionable repair.
- **Tool or transport defect:** a documented valid request fails, result schema
  is wrong, or the typed result cannot be consumed as declared.
- **Agent slip:** the contract and recovery path were clear; one isolated
  malformed call does not establish a product problem.

Compare a direct known-valid execution separately from the trace. A direct pass
proves the operation works; it does not erase caller friction. A trace with no
call is out of scope unless availability or adoption is under review.

## Measure benchmark friction

For comparable runs, freeze task, image, server revision, model, client,
budget, and MCP configuration. Report final verifier outcome separately from:

- first-call validity;
- operation inspection before execution;
- validation failures by field and invariant;
- recovery attempts and whether a retry used the returned error;
- typed-result use versus manual fallback; and
- added calls, tokens, and elapsed time.

Use repeated tasks or attempts before claiming a stable rate. A successful
answer after retries is evidence of recoverability, not evidence of a
frictionless interface.

## Recommend the smallest repair

Prefer, in order: clearer field metadata and examples; actionable errors;
schema-level constraints; a narrowly scoped adapter or helper. Normalize input
only when it preserves the documented mathematical meaning and does not hide
ambiguity, duplicate terms, or caller mistakes. Preserve bounded operation
semantics and return values directly.

Report confirmed behavior, the exact public evidence, the affected stage,
proportionate repair, and what remains untested. State when the evidence only
supports an agent slip rather than a tool-design finding.
