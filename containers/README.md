# Reusable evaluation runtimes

This image supplies the fixed Lean compiler used to replay bounded submitted
source in an independent verifier.

| Image | Purpose | Must not contain |
| --- | --- | --- |
| `jacobian-lean-checker` | Compile and inspect submitted Lean source in a clean verifier. | Task solution, task-specific expected output, or the REPL. |

The release workflow publishes this image to GHCR after a trusted `main` push.
It emits immutable commit tags, an OCI digest, SBOM, provenance, and a size
receipt. A Harbor task must pin the resulting `@sha256:` reference in its
environment profile or task Dockerfile. Tags are discovery conveniences, not
evaluation identity.

The checker image is intentionally Lean-only. A future task that genuinely
requires Mathlib must use a separately measured and published runtime; adding
Mathlib to this base would impose its multi-gigabyte footprint on every Lean
source replay.
