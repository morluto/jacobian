# Formal-dataset materialization

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`dataset.formal.materialize` converts one pinned MiniF2F or ProofNet row into a
content-addressed replay artifact. It is a deterministic ingestion boundary,
not a downloader, theorem executor, or verifier.

## Request contract

Every request supplies:

- the dataset ID, immutable revision, sample ID, and source URL;
- one dataset-specific row;
- the required Lean version, project source URL, and project revision;
- optional Mathlib revision;
- ordered imports, namespace, theorem context, and project-file digests; and
- an optional expected row digest for fail-closed source replay.

MiniF2F rows bind a declared `train`, `valid`, or `test` split. ProofNet rows
retain the source split label because published ports use different split
taxonomies. The registry is closed: unknown dataset IDs are rejected rather
than interpreted heuristically.

```json
{
  "capability_id": "dataset.formal.materialize",
  "input": {
    "dataset_revision": "3a5dceb842b916345a4d7bb7dc4c1dbd4b98aa",
    "sample_id": "mathd_algebra_1",
    "source_url": "https://huggingface.co/datasets/Tonic/MiniF2F",
    "row": {
      "dataset_id": "MINIF2F",
      "name": "mathd_algebra_1",
      "split": "test",
      "header": "import Mathlib",
      "formal_statement": "theorem mathd_algebra_1 : True := by trivial",
      "goal": "True",
      "informal_statement": "A fixture statement."
    },
    "environment": {
      "lean_version": "4.31.0",
      "project_source_url": "https://github.com/leanprover-community/mathlib4",
      "project_revision": "project-commit",
      "mathlib_revision": "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f",
      "imports": ["Mathlib"],
      "project_files": [
        {
          "path": "lean-toolchain",
          "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
      ]
    }
  }
}
```

## Replay and diagnostics

The domain-owned producer normalizes line endings, preserves all source
whitespace, preserves leading blank lines, and ensures at least one final
newline. Inputs must already be NFC-normalized so the adapter never rewrites
Unicode tokens inside Lean source. It binds separate SHA-256 digests for the
complete canonical source row, normalized Lean source, and declared execution
environment. The artifact retains the dataset split explicitly. Repeating the
same request produces the same normalized source, digests, and result artifact
URI.

As a standard domain operation, the capability response places the
materialized record under `output.result` and exposes its content-addressed
artifact as `output.result_uri`.

The output explicitly reports `NOT_EXECUTED`. Version mismatches and missing
project-file bindings produce typed diagnostics so a caller can provision the
correct checkout before execution. The adapter never silently falls back to
the host project.

## Trust boundary

Materialization is `COMPUTED` and always `UNVERIFIED`. It does not establish:

- that the source compiles in the declared project;
- that a proof is accepted;
- that the formal statement matches the informal statement; or
- that the theorem is true.

Completed source must be replayed through an applicable Lean execution path,
and theorem verification remains exclusively behind the independently
authorized `lean.check` boundary.
