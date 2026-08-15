# Author a Harbor benchmark task

[Documentation home](../index.md)

Harbor tasks are external evaluation assets. Choose a bounded, difficult
mathematical claim because it reveals a capability needed for serious
mathematical investigation, not because an existing Jacobian operation makes
it easy. A task can expose an operation gap; tool availability belongs in the
experimental treatment, not in the definition of a good task.

Put the task under its dataset root, keep agent-visible input separate from
Oracle-only solution material and verifier-only tests, and pin the environment
that evaluates it. The task's verifier owns correctness and any task-specific
witness needed for replay; Jacobian owns neither that verifier nor its data
lifecycle. For an atomic task, score the replayed mathematical predicate
binary: valid result and required witness is `1`, otherwise `0`. Keep tool use
and verifier diagnostics as separate observations; add partial credit only for
explicit independent replayable subclaims.

Use the repository's Harbor Make targets to plan and validate the exact task
you changed. Keep the task's score and interpretation distinct from Jacobian's
two-tool operation contract.
