# Harbor task template

Copy this directory into a registered dataset and replace every placeholder.
Keep `instruction.md` and `environment/` agent-visible; keep `solution/` and
`tests/` Oracle/verifier-only. Add task-specific schemas without weakening the
common submission envelope or assurance ceiling. Add one
verifier-owned `tests/public_contract.json` and generate the marked submission
block plus `environment/submission_schema.json` with the internal
`benchmarks.tooling.public_contract` command; do not hand-maintain duplicate
protocol declarations. Add one
`members/<task-id>.toml` record and run the exact leaf gate:
`make harbor-check-task DATASET=<dataset-id> TASKS="<task-id>"`. Run
`make harbor-oracle-task DATASET=<dataset-id> TASKS="<task-id>"` after the
contract gate passes. Use the full `make harbor-check` only when changing
shared Harbor tooling, schemas, registry, suite policy, or another
control-plane file. Create a snapshot lock only when freezing an intentional
evaluation or publication set; do not hand-edit or commit a dataset-root
`dataset.toml`.
