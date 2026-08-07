# Reusable evaluation runtimes

These images separate an agent/provider's exploratory toolchain from a
verifier's authority to replay a submitted mathematical artifact.

| Image | Purpose | Must not contain |
| --- | --- | --- |
| `jacobian-lean-checker` | Compile and inspect submitted Lean source in a clean verifier. | Task solution, task-specific expected output, or the REPL. |
| `jacobian-lean-repl-agent` | Run the pinned Lean REPL for provider-feasibility and agent environments. | Authority to certify a report as independently verified. |

The release workflow publishes both images to GHCR after a trusted `main` push.
It emits immutable commit tags, an OCI digest, SBOM, provenance, and a size
receipt. A Harbor task must pin the resulting `@sha256:` reference in its
environment profile or task Dockerfile. Tags are discovery conveniences, not
evaluation identity.

The checker image is intentionally Lean-only. A future task that genuinely
requires Mathlib must use a separately measured and published runtime; adding
Mathlib to this base would impose its multi-gigabyte footprint on every Lean
source replay.
