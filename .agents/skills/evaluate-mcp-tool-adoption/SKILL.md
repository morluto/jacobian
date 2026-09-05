---
name: evaluate-mcp-tool-adoption
description: Investigate MCP tool availability, discovery, and selection, including controlled adoption evaluations.
---

# Evaluate MCP Tool Adoption

Evaluate an agent-facing tool as a routing surface, not as a call-count target.
Distinguish availability, discovery, selection, and execution before changing
server instructions, tool metadata, or the operation contract. Load this skill
only for the evaluator; never enable it in an agent-under-test configuration.

Use `harbor-benchmarks` for a Harbor task or verifier change and
`verifier-evaluations` for its mathematical verifier. This skill owns only the
adoption diagnosis and its evidence boundary.

## Freeze the question

For a controlled comparison, write one falsifiable question, such as “does
server-level guidance cause an agent to inspect a matching operation when the
ordinary prompt does not?” Keep
the model, model version, reasoning effort, client version, Jacobian revision,
image digest, task digest, MCP configuration, egress/proxy state, attempt
count, and budgets fixed across comparable arms.

Record the relevant prompt and agent-visible MCP surface. In a controlled
comparison, do not include an operation ID, answer, or routing instruction in a
natural-use arm. Do not score a tool call as success: final mathematical correctness,
declared witness validity, and truthful claims remain the task outcomes.

## Establish the four-stage control matrix

Run deterministic checks before model calls, then use the smallest set of
model arms that distinguishes the suspected stage.

| Stage | Minimal probe | What a pass establishes |
| --- | --- | --- |
| Availability | Start the configured MCP server and inspect the initialization/tool list. | The client receives the intended connection, tools, descriptions, and schemas. |
| Discovery | Search with a natural user phrase and natural category; inspect only returned candidates. | A relevant operation is reachable without prior taxonomy or exact-ID knowledge. |
| Execution | Invoke the known matching operation directly with one frozen valid payload. | Input schema, dispatch, and typed result work. |
| Selection | Give a fresh agent the same mathematical task under no cue, server guidance only, and one clearly labeled task-level routing cue. | The effect of routing/salience is separable from connection and execution. |

Keep the execution control separate from discovery. A direct operation ID can
prove dispatch, but cannot prove that an agent would find that operation.
Likewise, a task-level instruction can prove that the client can call a tool,
but cannot prove that server initialization instructions are salient enough to
change ordinary selection.

## When model comparisons are needed

Use [model comparisons](references/model-comparisons.md) for controlled arms,
event capture, and scoring. Run only the arms needed to answer the frozen
question within user-authorized cost boundaries. A deterministic diagnosis does
not require a model experiment.

## Attribute conservatively

Use the smallest supported conclusion:

- A missing session, missing tool, or changed schema is an availability or
  presentation failure.
- A direct call succeeding while natural search misses the operation is a
  discovery/taxonomy problem.
- A direct-call control succeeding while uncued agents avoid the tool is a
  selection or salience observation, not proof that initialization guidance is
  absent from the model context.
- A task-level cue changing behavior proves that the cue can route behavior; it
  does not show that the call improved correctness or should become mandatory.
- Invalid typed input or output belongs to the operation contract or transport
  boundary, not to agent adoption.

Use at least one repeated or independently varied case before claiming a
stable effect. Open an issue only with the exact prompt, environment, event
evidence, and the distinction between verified behavior and the remaining
hypothesis. Keep outcome evaluation separate from telemetry intended to explain
tool selection.
