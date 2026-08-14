# Run agent evaluations

[Documentation home](../index.md) · [Evaluation reference](../reference/evaluations/evaluation-methods.md)

Agent evaluations are explicit operator-run Harbor experiments. Freeze the
task, model, environment, budget, repetitions, and treatment definition before
running them. Compare a no-Jacobian control with a treatment exposing only the
public `math.find` and `math.run` surface; do not add a prescribed tool-call
sequence or server-side workflow.

Validate the selected task with its Harbor workflow before spending model or
Oracle resources. Publish results with task and environment digests, measured
outcomes, and limitations. The resulting logs and scores are evaluation data;
they do not create Jacobian artifacts, workspaces, or verification records.

Run each arm in a fresh temporary `CODEX_HOME`, never through direct host `codex exec`.
The control must have no Jacobian MCP server; the treatment must expose only the
intended Jacobian MCP configuration and no Jacobian Skill.
