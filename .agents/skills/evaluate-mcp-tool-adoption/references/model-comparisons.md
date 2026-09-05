# Controlled model comparisons

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
