# Author a Harbor benchmark task

[Documentation home](../index.md)

Harbor tasks are external evaluation assets. Put the task under its dataset
root, keep agent-visible input separate from Oracle-only solution material and
verifier-only tests, and pin the environment that evaluates it. The task's
verifier owns correctness and any evidence it needs; Jacobian owns neither that
verifier nor its data lifecycle.

Use the repository's Harbor Make targets to plan and validate the exact task
you changed. Keep the task's score and interpretation distinct from Jacobian's
two-tool operation contract.
