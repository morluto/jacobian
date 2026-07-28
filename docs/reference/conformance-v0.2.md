# v0.2 conformance specification

[Documentation home](../index.md)

- Status: Normative release gate for `0.2.0a0`
- Release specification:
  [v0.2 capability-first mathematical operations](specifications/v0.2.md)

This document defines the complete normative conformance suite for Jacobian
v0.2. Search, evaluation, canonicalization, transformation proposal, and solver
output are untrusted unless an authorized checker accepts their bound evidence.

The agent-facing MCP surface consists of `capability.describe` and
`capability.invoke`. Operation names in this document, such as
`witness.verify`, are capability IDs rather than additional top-level MCP
tools. Clients may inspect the small default summary to judge fit, then request
`view = CONTRACT` (or the legacy `COMPACT` alias) for the installed
capability's validation-equivalent input schema, or `view = FULL` for the
complete descriptor and output schema. They then pass that ID and a
schema-valid payload to `capability.invoke`.

## Result invariants

Every v0.2 operation preserves the core result model:

- execution, input validity, mathematical conclusion, assurance, and evidence
  remain orthogonal;
- cancellation, timeout, malformed plugin output, and reached limits never
  imply a mathematical conclusion;
- search experiments and canonicalization results are always `UNVERIFIED`;
- invoking `transform.apply` or `polytope.separate` cannot create verification
  records;
- a capability may return `VERIFIED` only when its descriptor supports
  `VERIFY` and an authorized independent checker accepts the exact bound
  evidence.

## Artifact identity

| ID | Scenario | Required result |
| --- | --- | --- |
| `ART-001` | Insert the same canonical object twice | Same object digest; idempotent insertion |
| `ART-002` | Encode equivalent reduced rationals | Same canonical object digest |
| `ART-003` | Reuse identical payload bytes under another schema | Different object identity |
| `ART-004` | Reuse identical payload bytes under another semantics digest | Different object identity |
| `ART-005` | Modify a stored blob | Digest verification failure |
| `ART-006` | Insert duplicate JSON keys or a disallowed float | Input rejected |
| `ART-007` | Exceed artifact size or nesting limits | Bounded rejection without partial commit |
| `ART-008` | Change committed manifest metadata without changing its blobs | Integrity verification failure |

## Operational and mathematical state

| ID | Scenario | Required result |
| --- | --- | --- |
| `RES-001` | Evaluator times out | `execution = TIMEOUT`; no false conclusion |
| `RES-002` | Evaluator crashes | `execution = ERROR`; no verified record |
| `RES-003` | Enumeration reaches a declared limit | Coverage is not exhaustive |
| `RES-004` | Floating evaluation reports a positive margin | Result remains unverified |
| `RES-005` | Exact evaluator claims exhaustive coverage without checked evidence | Result remains unverified |

## Witness verification

| ID | Scenario | Required result |
| --- | --- | --- |
| `WIT-001` | Verify a valid direct witness | Verified logical effect |
| `WIT-002` | Reference an object outside the witness domain | Witness rejected |
| `WIT-003` | Bind a valid witness to another candidate | Binding rejected |
| `WIT-004` | Mutate one witness component | Replayed result changes or fails |
| `WIT-005` | Oracle returns a witness after timeout metadata | Witness is judged only by independent replay |

## Certificate verification

| ID | Scenario | Required result |
| --- | --- | --- |
| `CRT-001` | Replay a valid finite enumeration certificate | Verified conclusion |
| `CRT-002` | Copy the certificate to another claim | Binding rejected |
| `CRT-003` | Change its semantics, scope, candidate, or encoding digest | Binding rejected |
| `CRT-004` | Request an unregistered checker | Result remains unverified |
| `CRT-005` | Use a revoked checker for new verification | Verification denied by policy |
| `CRT-006` | Corrupt the certificate payload | Verification failure |
| `CRT-007` | Return `NONE_CERTIFIED` without a verified certificate record | Protocol violation |
| `CRT-008` | Authorize a checker with an empty schema or semantics allowlist | Authorization rejected; omission is not a wildcard |

## Plugin isolation

| ID | Scenario | Required result |
| --- | --- | --- |
| `PLG-001` | Plugin manifest declares a trusted checker | Declaration ignored or rejected |
| `PLG-002` | Plugin omits a required capability | Failure before execution |
| `PLG-003` | Plugin changes semantics without changing its digest | Manifest or digest validation failure |
| `PLG-004` | Verification package imports search implementation | Dependency-boundary test failure |
| `PLG-005` | Two domains expose different optional capabilities | Capability discovery and invocation remain usable in both |

## Shrinking

| ID | Scenario | Required result |
| --- | --- | --- |
| `SHR-001` | Reducer proposes a smaller preserving target | Accepted after checker replay |
| `SHR-002` | Reducer breaks the predicate | Rejected |
| `SHR-003` | Budget ends before checked neighborhood exhaustion | Honest minimality class |
| `SHR-004` | Capability result claims global minimality without a certificate | Protocol violation |
| `SHR-005` | Final target lacks a fresh verification record | Output remains unverified |
| `SHR-006` | Reducer returns no proposals after an accepted reduction | `NONE`; reducer silence does not establish local minimality |

## Record and replay

| ID | Scenario | Required result |
| --- | --- | --- |
| `RPL-001` | Inspect a verification record | Exact checker and environment digests are present |
| `RPL-002` | Change authorized checker bytes before replay | New verification is denied |
| `RPL-003` | Replay a completed bundle in a clean process | Same verified conclusion |
| `RPL-004` | Replay with a missing dependency artifact | Explicit resolution failure |
| `RPL-005` | Revoke a checker while verification is in flight | No record commits after revocation completes |
| `RPL-006` | Present schema-invalid data under an authorized schema URI | Verification is rejected before checker dispatch |

## Enumeration experiments

| ID | Scenario | Required result |
| --- | --- | --- |
| `ENUM-001` | A finite enumerator reports its complete declared scope | Durable `COMPLETED` snapshot, `COMPLETE` stop reason, exhaustive coverage, auditable scope and archive, but `UNVERIFIED` |
| `ENUM-002` | `candidates_max` is reached first | `CANDIDATE_LIMIT`, bounded coverage, no exhaustive or verified conclusion |
| `ENUM-003` | The wall-clock budget is reached first | `TIMEOUT`, `WALL_TIME_LIMIT`, bounded coverage |
| `ENUM-004` | Cancellation is requested while work is pending or running | Cooperative `CANCELLED`; already committed artifacts remain readable |
| `ENUM-005` | Cancellation targets a terminal experiment | `accepted = false`; immutable terminal state is unchanged |
| `ENUM-006` | An incomplete page has no cursor, a complete page has a cursor, a cursor does not advance, or scope changes between pages | Experiment ends in `ERROR`; no completeness claim |
| `ENUM-007` | A plugin returns more candidates than the requested page size | Experiment ends in `ERROR`; accounting remains internally consistent |
| `ENUM-008` | Quotienting is requested from a plugin without a canonicalizer | Rejected before the experiment starts |
| `ENUM-009` | A process restarts with pending or running local experiments | Orphaned experiments become inspectable `ERROR` records; committed artifacts remain |
| `ENUM-010` | An enumerator returns a candidate that violates the installed candidate schema | Experiment ends in `ERROR`; the invalid candidate is not archived |

An exhaustive search snapshot records what the untrusted enumerator reported.
It is not a proof that the scope is mathematically complete. A theorem derived
from absence of candidates requires a separately checked enumeration
certificate format.

## Canonicalization

| ID | Scenario | Required result |
| --- | --- | --- |
| `CAN-001` | Two isomorphic structures are canonicalized by the same implementation | Same canonical object and implementation-bound canonical key |
| `CAN-002` | Two non-isomorphic structures are canonicalized | Distinct keys for the bundled reference canonicalizer |
| `CAN-003` | Canonicalizer implementation bytes change | Resolution fails or the canonicalizer-bound key changes |
| `CAN-004` | Canonicalizer output is malformed or violates the candidate schema | Rejected or operational error; never a verified result |
| `CAN-005` | A faulty canonicalizer collides two candidates | Search may deduplicate them, but the archive and completeness remain `UNVERIFIED` |

Canonical artifact identity represents the mathematical object. The separate
canonical key binds that object digest to the implementation digest used to
derive it.

## Representation transformations

| ID | Scenario | Required result |
| --- | --- | --- |
| `TRF-001` | A transformer proposes an equivalence, over-approximation, under-approximation, or heuristic relation | Bound target, relation, obligation, and transformer digest; result remains `UNVERIFIED` |
| `TRF-002` | An authorized checker independently replays a valid relation | `VERIFIED` result and immutable transformation-verification record |
| `TRF-003` | Source or target is substituted | Rejected; no verification record |
| `TRF-004` | Relation label is strengthened or changed | Rejected; no verification record |
| `TRF-005` | Obligation payload is changed without its digest | Contract validation rejects the envelope |
| `TRF-006` | Checker format, schema, source semantics, or target semantics is incompatible | Fail-closed checker-selection rejection |

## Exact finite-polytope operations

The v0.2 backend accepts finite rational generator sets in V-representation.
It supports exact coordinate projection, convex-hull membership, and strict
linear separation. H/V conversion, general implicit polytopes, redundancy
removal, and facet/incidence enumeration remain later backend work.

| ID | Scenario | Required result |
| --- | --- | --- |
| `POLY-001` | A point is in the finite convex hull | Exact convex-combination witness, initially `UNVERIFIED`; independent replay may verify it |
| `POLY-002` | A point is outside the finite convex hull | Primitive exact rational separator certificate, initially `UNVERIFIED`; independent exhaustive replay may verify it |
| `POLY-003` | Separator coefficients, right-hand side, values, or margin are changed | Independent checker rejects, even if the envelope payload digest is recomputed |
| `POLY-004` | Projection is requested | Derived point and generator-set artifacts bind their source artifacts |
| `POLY-005` | Dimensions, projection indices, or canonical rationals are invalid | Input rejected before solver dispatch |
| `POLY-006` | Z3 returns unknown, times out, or fails operationally | Mathematical conclusion remains `UNKNOWN`; no decisive evidence |

Z3 is an evidence generator, not part of the independent checker. The bundled
checkers use Python `Fraction` arithmetic and do not import the solver or search
implementation.

## Public adapters

| ID | Scenario | Required result |
| --- | --- | --- |
| `ADP-001` | List MCP tools using SDK `2.0.0b2` | Exactly `capability.describe` and `capability.invoke`; installed mathematical operations are capability IDs, not top-level tools |
| `ADP-002` | Read `experiment://<id>` | Latest durable snapshot |
| `ADP-003` | Read experiment accounting, scope, or archive resources | Compact metadata and artifact handles; large pages remain artifact resources |
| `ADP-004` | Describe and invoke an installed capability through MCP | The descriptor schema is enforced and the result preserves execution, assurance, evidence, artifact, and provenance fields |
| `ADP-005` | Invoke the same capability through CLI or Python | Same capability contract, validation, and assurance labels |

## Cross-domain gate

The same experiment contracts must enumerate both the graph/path and integer
matrix reference domains. The matrix plugin deliberately has no graph
canonicalizer; requesting graph-style quotienting from it must fail cleanly.
The row-major matrix transformation and finite-polytope evidence must use the
same artifact, checker registry, result, and verification-record substrate as
the verification capabilities.

## Release gate

v0.2 is conformant only when:

1. every applicable case above has a behavioral test;
2. both reference plugins expose capability IDs that complete verification and
   bounded-enumeration operations through `capability.invoke`;
3. one transformation and one exact separator replay to `VERIFIED`;
4. cancellation and reached limits remain inspectable and unverified;
5. the CLI, MCP adapter, built wheel, and npm launcher expose equivalent
   release versions.
