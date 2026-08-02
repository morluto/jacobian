# Harbor task template

Copy this directory into a registered dataset and replace every placeholder.
Keep `instruction.md` and `environment/` agent-visible; keep `solution/` and
`tests/` Oracle/verifier-only. Add task-specific schemas without weakening the
common submission envelope or assurance ceiling. Add one
`members/<task-id>.toml` record and run `make harbor-check` to validate the
bundle and Harbor's task checksum. Create a snapshot lock only when freezing an
intentional evaluation or publication set; do not hand-edit or commit a
dataset-root `dataset.toml`.
