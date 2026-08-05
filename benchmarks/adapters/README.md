# Harbor source adapters

This directory is reserved for reproducible conversions from pinned external
benchmark sources into task bundles under the owning
`benchmarks/datasets/<dataset>/` Harbor dataset root. An
adapter must record the immutable source revision and content digest, license
and redistribution status, included and excluded rows, deterministic conversion
command, pinned dependencies, Oracle evidence, and parity evidence when it
claims equivalence to the source.

Manually authored or substantially transformed tasks remain authored Harbor
tasks with provenance metadata; an “inspired by” citation is not an adapter.
Internal task-model and adapter-infrastructure fixtures belong under
`benchmarks/validation/fixtures/`, not in this contributor-facing directory,
and are not examples to copy when adding an external source.

Follow Harbor's maintained adapter layout: a locked Python package and CLI
generate ordinary task directories containing `task.toml`, `instruction.md`,
`environment/`, `solution/`, and `tests/`. Reuse an existing Harbor mapper when
the source already follows Terminal-Bench or another supported format; keep
source-specific code to the conversion that cannot be expressed by that
mapper. Do not introduce a Jacobian-only task format or generate a root
`dataset.toml`.

Each adapter directory must contain `source.lock.json`, `README.md`, a locked
generator package, and an executable parity/regeneration check. The lock conforms to
`benchmarks/schemas/source-adapter-lock.schema.json` and binds source revision,
license, row selection, dependencies, output task digests, Oracle evidence,
and parity evidence. `make harbor-check` validates every lock without network
access; `make harbor-adapter-check ADAPTER=<id>` additionally runs that
adapter's deterministic regeneration check.

Parity claims follow Harbor's ABC-Bench example: freeze the source selection,
agent and model versions, environment, repetitions, metric, and raw result
digests; report Oracle coverage separately; and count verifier or infrastructure
failures explicitly rather than smoothing them into benchmark misses. Generated
tasks remain directly runnable with Harbor's `-p` path and task filters.
