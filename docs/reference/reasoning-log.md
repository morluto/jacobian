# External reasoning-log protocol

[Documentation home](../index.md) · [Capability surface](tools.md)

Jacobian requires MCP agents to keep a concise external work log. Entries are
model-authored summaries, not hidden or internal chain-of-thought. They remain
operational and unverified and never change mathematical assurance.

## Lifecycle

1. Call `reasoning.write` with `phase: "PLAN"`; retain its `run_id`.
2. Before each capability execution, write `BEFORE_TOOL` with that run ID,
   exact capability ID, mode, and a short purpose. Retain the returned `call_id`.
3. Add both IDs to the matching `math.run` request.
4. After every result, write `AFTER_TOOL` with `interpretation_status:
   "INTERPRETED"`, the same IDs, a concise interpretation, and the reported
   execution status, assurance level, and completeness status. If the result
   content was lost after a response failure or runtime restart, use
   `interpretation_status: "RESULT_UNAVAILABLE"` and omit those reported fields.
5. Write one `FINAL` audit after all calls have been interpreted.

One run executes only one capability cycle at a time and accepts at most 64
cycles. Different runs and tenants remain concurrent. A current runtime never
declares a call interrupted because of elapsed time. After a runtime restart, an
explicit `RESULT_UNAVAILABLE` entry atomically records the old call as an `ERROR`
non-conclusion before closing its interpretation cycle.

Each summary is limited to 512 Unicode characters and 2048 UTF-8 bytes. Do not
include prompts, credentials, personal data, raw request payloads, or copied tool
output. The server stores a canonical request digest and result digest plus the
actual capability ID/version/mode, execution status, assurance, completeness,
scope digest, diagnostics, artifact URIs, and episode URI.

Read the durable append-only log at `reasoning://run/{run_id}`. It is returned as
newline-delimited JSON. Events carry a contiguous sequence number, and SQLite
rejects event updates and deletes. Canonical request and result digests bind calls
without claiming protection against a privileged state-database operator.

## Enforcement and recovery

`REQUIRED` rejects an unbound, mismatched, reused, or out-of-order invocation
before mathematical execution. `AUDIT` records protocol violations but permits
legacy unbound invocations. `OFF` removes `reasoning.write`, its resource, and
the extra invocation fields.

A duplicate or concurrent use of one `call_id` never starts a second
calculation. If a response is lost after completion, inspect the log rather than
blindly retrying. Use `RESULT_UNAVAILABLE` when the actual result content cannot
be recovered, and reserve a new call only when recomputation is genuinely
required. V1 assumes one server runtime owns a tenant state directory; multiple
MCP clients may share that runtime, and run/call IDs are bearer identifiers inside
the tenant.

Jacobian cannot prevent an agent from emitting a final answer outside MCP. A run
without `FINAL` is therefore auditable as incomplete, rather than falsely treated
as completed. Timestamps and append order constrain post-hoc insertion but cannot
prove that a summary accurately represents a model's private cognition.

The log follows the tenant state directory's retention and access controls. V1
does not add per-entry deletion, automatic expiry, listing, or encryption at rest.
The run resource is its JSONL export. Delete or rotate the complete tenant state
directory to remove logs. Evaluation jobs must use one ephemeral tenant state
directory per trial, export required evidence, and delete the directory after the
declared retention period. Operators handling persistent or sensitive research
must protect and rotate the complete tenant state directory.
Request and result digests are integrity bindings, not anonymization: an operator
with the log may be able to guess low-entropy inputs or outputs offline and compare
their hashes. Do not use sensitive payloads in a tenant whose state is not trusted.

## Evaluation use

For weak-model studies, run paired jobs with the same task set, model, prompt,
tool portfolio, and sampling settings. Compare `OFF` with `REQUIRED`; use
`AUDIT` only to measure naturally occurring protocol violations without blocking
the job. Report task correctness and false-certification rate separately from
protocol adherence. The transcript telemetry records phase counts, bound calls,
missing interpretations, run identity, and total summary characters, but not the
summary text.

Useful secondary measures are capability-selection accuracy, completed
tool cycles, abandoned calls, final-audit rate, token and latency overhead, and
the fraction of summaries that merely restate tool status. For a stronger-model
score `S`, weak baseline `W_off`, and weak treatment `W_required`, report the
closed performance gap as `(W_required - W_off) / (S - W_off)` only when the
denominator is positive, alongside raw scores and confidence intervals. Logs are
diagnostic covariates, not evidence that the protocol caused a mathematical
improvement; causal claims require randomized or counterbalanced paired runs.
