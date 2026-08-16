# Harbor task template

Copy this directory into a registered dataset and replace every placeholder.
Keep `instruction.md` and `environment/` agent-visible; keep `solution/` and
`tests/` Oracle/verifier-only. Start with a typed `result`; add a finite,
task-specific `witness` only when replay genuinely needs external mathematical
data. Never add generic assurance, scope, completeness, limitations, or prose
explanation fields. Add one verifier-owned `tests/public_contract.json` and
generate the marked submission block plus `environment/submission_schema.json` with the internal
`benchmarks.tooling.public_contract` command; do not hand-maintain duplicate
protocol declarations. Add one
`members/<task-id>.toml` record and run the exact leaf gate:
`make harbor-prepare-task DATASET=<dataset-id> TASKS="<task-id>"` then
`make harbor-validate-task DATASET=<dataset-id> TASKS="<task-id>"`. Use the
full `make harbor-check` only when changing
shared Harbor tooling, schemas, registry, suite policy, or another
control-plane file. Create a snapshot lock only when freezing an intentional
evaluation or publication set; do not hand-edit or commit a dataset-root
`dataset.toml`.

The result represents a mathematical value, not a preferred JSON or textual
rendering. Normalize and compare equivalent values unless the task explicitly
evaluates canonicalization; declare any such ordering or normal-form rule in
the public contract. Do not use `answer.txt` as an answer channel. Keep it only
as non-authoritative source material, if it is needed at all.

The verifier must replay the claim from frozen `input.json`. Do not score a
hidden `tests/expected.json` field, a lowest-terms rendering, a formula
string, or keyword-bearing prose. Do not default a witness path to
`evidence/answer.txt`. Do not copy a universal certificate union into a
generated family; each task's schema admits only the certificate that family
can reward. Instruction, schema, solver, and verifier must accept the same
objects. See [authoring a Harbor benchmark task](../../../docs/how-to/author-harbor-benchmark-task.md)
and [benchmark contracts](../../../docs/reference/evaluations/benchmark-contracts.md).
