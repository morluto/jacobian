---
name: evaluate-mcp-tool-adoption
description: Diagnose whether agents can see, discover, select, and correctly execute MCP tools without conflating those stages. Use when investigating why agents did or did not use an MCP tool, evaluating MCP server instructions or tool descriptions, testing natural-language discovery, or designing a frozen control/treatment adoption evaluation. Do not use for ordinary MCP implementation or task-specific benchmark authoring.
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

Write one falsifiable question, such as “does server-level guidance cause an
agent to inspect a matching operation when the ordinary prompt does not?” Keep
the model, model version, reasoning effort, client version, Jacobian revision,
image digest, task digest, MCP configuration, egress/proxy state, attempt
count, and budgets fixed across comparable arms.

Record the public task prompt and the complete agent-visible MCP surface. Do
not include an operation ID, answer, or routing instruction in a natural-use
arm. Do not score a tool call as success: final mathematical correctness,
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

## Run clean model arms

Use fresh contexts and identical task inputs. Change one intervention at a
time:

1. **No cue:** expose the MCP server with the ordinary task prompt.
2. **Server guidance:** change only the initialization guidance or tool
   presentation being evaluated.
3. **Task-level cue:** state a short conditional lookup request in the task;
   label this as an intervention, never as natural behavior.
4. **Direct-execution control:** name the matching operation and require one
   typed execution while forbidding a manual substitute; use this only to
   localize a suspected availability or execution failure.

Avoid a routing cue that mandates a result, a proof strategy, or routine calls
when no operation fits. If a matching operation is materially cheaper than a
fallback, document why with a bounded representative input; otherwise non-use
may be rational rather than a usability defect.

## Capture and score evidence

Persist raw initialization, tool-list, `math.find`, inspection, `math.run`,
and final-answer events. Record operation IDs, request payloads, typed results,
retries, visible MCP bytes, input/output tokens, elapsed time, task reward,
and verifier outcome. Hash frozen task and configuration inputs.

Report these separately:

- final mathematical outcome and verifier result;
- discovery of an applicable operation;
- valid first execution, recovery, and use of the typed result;
- resource cost relative to the no-cue arm; and
- failures by stage, not by a generic “tool use” label.

Do not infer hidden reasoning from a transcript. A successful answer without a
call is valid task success, and a call without a correct answer is not adoption
success.

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
