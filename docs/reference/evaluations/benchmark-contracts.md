# Benchmark contracts

[Documentation home](../../index.md) · [Tool surface](../tools.md)

All executable benchmark cases are Harbor tasks under their dataset roots
([`benchmarks/datasets/`](../../../benchmarks/README.md)). The datasets
retain separate claims:

- `mathematical-benchmarks-v1` checks fixed workflows and evidence handling;
- `symbolic-coordination-v1` owns the exact polynomial-map coordination pilot;
- `public-reproductions-v1` replays known public mathematical outcomes;
- `conjecture-probes-v1` checks independently replayable bounded conjecture progress;
- `research-diagnostics-v1` supports answer-visible case diagnostics;
- `provider-feasibility-v1` reproduces optional-provider pins and outcomes;
- `examples-v1` owns non-comparative tutorial and smoke workflows.

`registry.toml` is the discovery index. A dataset's `suite.toml` owns its
stable policy, while sorted `members/*.toml` records authoritatively bind
canonical task IDs to provenance, assurance, environment, and verifier
contracts. Content-addressed snapshot locks freeze intentional evaluation sets;
publication manifests are generated from locks outside dataset roots.

Dataset identity is a claim boundary: model observations, public
reproductions, answer-visible research diagnostics, runtime measurements,
provider feasibility, and examples must not share an interpretation merely
because they use one task format.
Subject taxonomy is separate from that boundary. Every member carries a
controlled `primary_domain` and a detailed `field`; these values organize the
portfolio but never rank tasks or prescribe a workflow.

The ownership boundary is deliberate. `benchmarks/datasets/` contains
executable Harbor cases and task-owned analysis records, while
`benchmarks/tooling/` contains reusable Harbor infrastructure. Analysis records
may capture discovery context, but they do not duplicate tasks, become Harbor
job input, or enter an agent container.

## Task contract

Tasks use `benchmarks/datasets/<dataset>/<task-id>/` with a maintainer README,
agent-visible instruction and environment, Oracle-only solution, and
verifier-only tests. Mathematical tasks use `mathematical-sciences`; runtime,
provider, and product-surface tasks use `software-systems`.

Every task has frozen agent-visible input, schema 1.4 metadata, an Oracle-only
solution, and a separate clean-room verifier. The common submission envelope
separates the conclusion, task-specific result, assurance, scope, completeness,
digest-bound evidence, optional verification record, and limitations. Unknown
fields fail closed. Task-specific schemas may narrow the result but cannot
weaken the envelope.

### Reusable evaluation images

Use a shared image only when its toolchain is part of the task's reproducible
runtime. The image digest, platform, toolchain version, and task digest together
identify the executable evaluation; a tag is only a human-facing discovery
label. Task Dockerfiles and environment profiles therefore use
`name@sha256:<digest>`, never `main`, a version tag, or an unpinned base.

Keep agent/provider images separate from verifier images. An agent image may
contain an exploratory service such as Lean REPL. A verifier image contains
only the independently needed checker and its pinned dependencies, then
replays a submitted artifact. Provider telemetry, tactic traces, and a
successful agent-side process are useful diagnostics but do not authorize a
mathematical conclusion.

The controlled `main`-only image workflow publishes the reusable Lean bases:

- `ghcr.io/morluto/jacobian-lean-checker` for Lean-source replay; and
- `ghcr.io/morluto/jacobian-lean-repl-agent` for the pinned provider runtime.

Publication records the immutable image digest, source revision, platform,
SBOM/provenance attestations, and unpacked image size. Measure runtime writable
storage separately in the actual Harbor runner before setting `storage_mb`:
image-layer size and writable-layer accounting are different quantities. Update
an image pin only in a deliberate task-contract change, recompute prospective
task digests, and rerun the selected Oracle.

## Task and verifier validation

Task and verifier validation is separate from model observation. For an
ordinary leaf change, validate the selected task and its exact Oracle:

```sh
make harbor-check-task DATASET=<dataset-id> TASKS="<task-id>"
make harbor-oracle-task DATASET=<dataset-id> TASKS="<task-id>"
```

The focused commands require explicit task IDs and do not fall back to all
tasks. The repository control-plane gate checks `registry.toml`, suite headers
and policy, shared contracts, schemas, global task-ID uniqueness, adapters, and
execution configuration. Add the explicit full-host gate only when the shared
verifier harness changes or a portfolio-wide reproduction is intended:

```sh
make harbor-check
make harbor-check-all
make benchmark-inventory OUTPUT=/tmp/benchmark-inventory.json
make harbor-oracle DATASET=mathematical-benchmarks-v1 FULL=1
```

A task README or a host-side regression under `benchmarks/validation/` changes
documentation or deterministic validation only, so it does not require an
Oracle. Changes to a task's executable input, environment, solution, member
record, or clean-room verifier do require the selected-task Oracle after the
contract check.

The suite module checks that each member ID names a direct Harbor task bundle
and validates the generated task digests. The verifier scores only evidence its
contract authorizes. Wrong mathematical answers, malformed or escaped evidence,
incomplete scope, and false certification receive zero reward. An Oracle answer
does not authorize `VERIFIED`.

### Diagnostics versus aggregate reward

Harbor reads multi-key `reward.json` and treats named floats as independent
metrics. Jacobian verifiers therefore use a **two-layer** scoring contract:

1. **Diagnostics** — report independent 0/1 scores such as `correctness`,
   `evidence_validity`, `scope_accuracy`, `assurance_calibration`, and
   `false_certification`. A zero aggregate must not erase which dimensions
   passed: invalid evidence with correct mathematics still reports
   `correctness = 1.0` and `evidence_validity = 0.0`.
2. **Aggregate `reward`** — the primary pass signal. Mandatory protocol
   dimensions are **non-compensable hard gates**. Soft weighted sums that still
   award most of the reward when a mandatory dimension fails are forbidden for
   evidence-bound tasks.

Mandatory hard gates for standard digest-bound tasks: protocol/envelope,
mathematical correctness, evidence binding, required scope, and false
certification. Assurance may be a hard gate or a documented soft partial
(`soft_assurance=True` in template `aggregate_reward`) only after every hard
gate passes.

**Forbidden leaky template:**

```python
# FORBIDDEN: evidence failure still yields reward ≈ 0.9
reward = (
    0 if not correct or false
    else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
)
```

Use template `aggregate_reward` (or an equivalent min-gate of mandatory
dimensions). Shared validation keeps an inventory ratchet so the leaky pattern
cannot reappear.

Each separate verifier owns its local `tests/verifier_support.py`; Harbor's
whole-task digest binds that copy, so validation does not synchronize it with a
global runtime helper. New tasks inherit the template copy, while shared fixes
are explicit migrations over selected tasks. Use the scoped `harbor-sync`
command only after such a deliberate update. Evidence has no arbitrary byte
cap, but its schema, digest, path, and workspace binding remain mandatory.
Verifier regression fixtures should also prove that malformed submissions do
not crash: exercise booleans where integers are expected, non-finite JSON
numbers, unhashable nested values, wrong-shaped input, wrong evidence digests
(aggregate reward zero while diagnostics stay independent), and assurance or
protocol failures whose independent diagnostics remain visible. A full Oracle
reward does not replace these negative-path checks.

`TIMEOUT`, `CANCELLED`, `ERROR`, incomplete enumeration, and failure to find a
witness remain non-conclusions. Only operator-authorized independent checkers
may accept `VERIFIED`.

### Jacobian verification records are not task verdicts

A Jacobian verification record is reusable, digest-bound evidence for one
typed operation; it is not a substitute for the Harbor task's clean-room
verifier. A task that accepts a record as evidence must publish the required
record fields, semantics identity, input and candidate bindings, checker
identity, scope, and evidence-path/digest rules, then validate those rules in
its own verifier. A record alone never proves a task's broader mathematical
claim, completeness requirement, or permitted assurance level.

For ordinary inline exact replay, Jacobian retains a verification record and
the semantics artifact it binds, while the input and candidate remain inline.
Evaluation authors must not require synthetic input/candidate artifacts merely
to make a task verifier consume this evidence. Conversely, a clean-room task
verifier must independently check the task relation or the record's declared
binding; it must not award credit solely because an agent reports `VERIFIED` or
supplies an opaque `artifact://` URI.

## Reproducible handoff

Record the git tree, suite and task digests, provider/runtime profile, model and
prompt settings, raw trace location, validation actually run, unresolved proof
obligations, and next action. Publishing a local dataset to a Harbor registry
requires separate authorization.

Ordinary executable task additions are leaf-only: the direct task bundle and
its matching `members/<task>.toml` record. They change the prospective suite
digest without rewriting stable suite policy or existing snapshot locks.
Intentional evaluation and publication events create a content-addressed lock
under `benchmarks/snapshots/`; publication manifests are generated under
ignored `dist/harbor/` from that lock.

See [evaluation methods](evaluation-methods.md) for workflow observation,
performance measurement, and interpretation guidance.

### Dataset migration

The former `agent-workflow-v1` corpus is now
`mathematical-benchmarks-v1`, with registry ID
`jacobian/mathematical-benchmarks-v1`. Task IDs remain stable. Five known
reproductions—`balanced-row-permutation`,
`closed-set-distance-strengthening-audit`, `coin-process-potential`,
`cyclic-vector-inequality`, and `superposition-proof-replay`—are members of
`public-reproductions-v1`. No active compatibility alias exists. The historical
`benchmarks/snapshots/agent-workflow-v1/` lock and ignored result paths are
preserved as historical evidence.
