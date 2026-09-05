# Harbor packaging details

## Older contracts

`harbor-prepare-task` is the scoped mutating step: it formats selected task
Python and refreshes public-contract and verifier checksums. For older contracts
that the modern prepare workflow does not support, use the dataset's own check
path and scoped `tools/sync_harbor_verifier_support.py`; do not rewrite a task
only to satisfy the helper.

## Task-local support

Treat task-local `tests/verifier_support.py` copies as authoritative because
Harbor's separate verifier image needs them in its build context. Migrate copies
explicitly, inspect their diffs, and refresh only their Dockerfile checksum
labels. Do not silently synchronize all tasks from a global runtime helper.
`verifier_bundle_checksum()` hashes filenames, NUL separators, and bytes; do
not concatenate `sha256(verifier.py)` with `sha256(verifier_support.py)`.
Refresh selected tasks with `make harbor-prepare-task` or scoped
`tools/sync_harbor_verifier_support.py`. Gold witness descriptors must resolve
to regular files under a non-symlink `solution/` root.

Create a snapshot only for an intentional evaluation or publication boundary:

```sh
make benchmark-snapshot DATASET=<dataset>
make benchmark-snapshot-validate LOCK=benchmarks/snapshots/<dataset>/<digest>.lock.json
```

Do not create or retain a snapshot merely because task contracts changed.
