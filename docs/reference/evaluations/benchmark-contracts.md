# Benchmark contracts

[Documentation home](../../index.md) · [Tool surface](../tools.md)

Harbor benchmarks are external evaluation products, not Jacobian runtime
features. A task owns its agent-visible instruction, hidden Oracle material,
verifier, pinned environment, and task digest under `benchmarks/datasets/`.
The evaluation author owns task selection, scoring, and any independent
validation needed by that task.

Choose tasks for the difficult mathematical capability they measure—especially
capabilities that matter for higher-level conjecture work—not because the
current operation library already solves them. A task that exposes an operation
gap is useful evidence. Jacobian availability is a treatment condition that an
experiment can compare with a tool-free control; it is not a criterion for
admitting a benchmark task.

Keep the boundary sharp: a benchmark may call `math.find` and `math.run`, but
it must pass complete bounded inputs and use returned typed values. It must not
assume a Jacobian workspace, artifact URI, persistence layer, checker registry,
or verification record. Freeze task and environment digests, keep hidden
material out of the agent environment, and report the task's limitations with
its result.

Experiments expose the public MCP surface they measure. The agent decides
whether to use Jacobian, whether discovery is useful, which operation to run,
and when to stop; a control condition may expose no Jacobian server. Do not add
product-facing runtime state machines, alternate assurance schemas, or
lifecycle APIs for a study. A reusable benchmark adapter stays a small consumer
of Harbor trial artifacts and the public MCP surface, never a second planner.

## Task outcome and diagnostics

For an atomic mathematical task, the default outcome is binary: the verifier
returns `1` only when it can replay a valid submitted result and every declared
task-specific witness condition; otherwise it returns `0`. Tool calls, prose,
confidence claims, and diagnostic observations do not earn credit. Report
input binding, witness validity, tool use, cost, and failure modes separately.
Use non-binary scoring only for a public task deliberately decomposed into
independent, meaningful, replayable mathematical subclaims.

The public submission shape is normally `{ "result": ... }`. Add
`"witness"` only for a finite task-specific mathematical object that replay
cannot derive from the frozen input and result; prefer structured certificate
data in `result`, and never require a duplicate result file or natural-language
explanation. Generic assurance, scope, completeness, limitation, and
verification-record fields are not part of ordinary mathematical submissions.

### Replay authority

The scored predicate is a task-local replay from the frozen verifier copy of
`input.json` and the submitted mathematical value. Hidden `tests/expected.json`
is an Oracle regression fixture, not the implementation of correctness.
Changing only the expected fixture must not change reward for a fixed
input/submission pair; a corrupted expected file must fail Oracle validation
instead of silently rewriting the mathematical truth.

Public notes that say the verifier replays a predicate must be true of the
verifier. Snapshot-label or registry-selection tasks must say so in the
instruction and schema; they must not advertise independent mathematical
replay while comparing a hidden label.

Independent propositions in one result are derived independently. A corrupted
claimed image, incomplete search, or one-sided certificate does not decide a
different collision, invertibility, or completeness claim.

### Mathematical representations

The typed result represents a mathematical object, not a preferred rendering of
one. Parse and compare the represented value:

- structured rationals with integer numerator and positive denominator,
  compared as `Fraction`, including unreduced encodings such as `2/8`;
- unordered collections (sets, maps, distributions, sparse polynomials) after
  parsing into the actual mathematical type, not as privileged list order;
- formulas as the smallest task-owned type (coefficients, an AST, an enum),
  not as a scored string literal;
- conclusions as finite reason/status enums the verifier already derives, not
  as keyword, length, or negation-regex checks over prose.

Do not encode an exact rational as one undifferentiated `"2/8"` string. Decimal
strings are allowed only as bounded components of a structured object when
JSON numbers are not interoperable. A task may require a canonical form only
when canonicalization is itself a stated mathematical outcome; the public
instruction and schema must then declare that requirement and its exact rule.

JSON Schema cannot express coprimality. If the instruction demands lowest
terms while the verifier normalizes `Fraction`, the public protocol is false.
Keep the equivalence-aware verifier and describe that policy.

Reject booleans as integers, zero or negative denominators under the stated
sign convention, non-finite numbers, and resource-bound violations. Those
checks are boundary validation, not rendering rules.

### Public protocol agreement

The instruction, schema, hidden solver, and verifier must accept the same
objects. A schema-valid, instruction-conforming submission must be able to
receive full reward. Do not leave `limitations` required in prose and forbidden
in the verifier, or declare `text/plain` for a JSON object.

Generated task families specialize the public schema to the claim family and
frozen case type. Certificate kind is public protocol, not a hidden answer.
Do not publish a universal `oneOf` of inverse, Keller, collision, grid, and
timeout certificates on every task. Where verdict and certificate kind are
coupled by the public claim, express the smallest family-specific union.

### Witness artifacts

`answer.txt` is never the authoritative submission interface. A task may retain
human-readable text as non-authoritative source material, but its hidden gold
solution and agent submission use the same structured contract. A witness file
is justified only when the verifier needs an external finite object for replay;
it must not duplicate the typed result or carry a prose explanation.

Do not require a witness whose only accepted content is a hash of `result`, a
JSON envelope equal to `result`, selected copied fields, boilerplate keywords,
or arbitrary nonempty text. Do not score a digest-bound file the verifier never
opens. The template helper `witness_list_is_bound` has no default
`evidence/answer.txt` path; pass an explicit path only for a declared artifact.

## Generated output

Task bundles and member records are reproducibility anchors and remain tracked.
Create an immutable snapshot lock only at an intentional evaluation or
publication boundary. Run outputs, collected traces, reports, and other
regenerable evidence belong under ignored `benchmarks/results/` or an external
artifact store. Do not mix generated study output with a runtime or benchmark
source refactor; publish it separately when it is needed for review.
