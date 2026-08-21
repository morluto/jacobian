# Lean source checking and Mathlib declaration search

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`lean.check` elaborates one bounded Lean source snippet in the fixed Lean 4.31.0
and pinned Mathlib Lake environment included in the service image. The snippet
may `import Mathlib`. It returns either `ELABORATED` or `REJECTED` and a bounded
list of typed diagnostics.

Each invocation uses a request-scoped temporary directory, removes it when the
process exits, and returns the resulting diagnostics. A timeout or process
failure is an execution failure, not an elaboration result.

`lean.declarations.search` searches the complete declaration environment of
the pinned Mathlib `v4.31.0` revision. It accepts a literal case-sensitive name
substring, up to four exact type-constant names, optional namespace and
declaration-kind filters, and a result limit of at most 20. Type constants and
namespaces use a conservative ASCII dotted-name grammar; the adapter passes
them as JSON data and never inserts them into Lean source.

Each match contains the exact declaration name, an 8,000-character
pretty-printed type preview with an explicit truncation flag, its kind, and its
defining module. Matching uses the complete Lean type, never the preview.
`EXHAUSTED` means the fixed environment was completely scanned;
`RESULT_LIMIT` means more matches may exist. The projection is for discovery:
pass a candidate name into a source snippet and use `lean.check` to validate the
actual proof term.

The repository-local environment is installed with:

```sh
make setup-lean
```

Source checkouts discover it under `lean/`. Other deployments can set
`JACOBIAN_MATHLIB_ROOT` to a directory containing this repository's pinned
`lake-manifest.json`, toolchain, and Mathlib cache. Missing, changed, timed-out,
or resource-limited environments return typed execution outcomes and never a
negative declaration-search conclusion.
