# SAT artifact contracts

[Documentation home](../../../index.md)

- Status: Experimental pre-stable contract
- Installed operation: `sat.cnf.materialize`
- Optional operations: `sat.model.find` and `sat.unsat_proof.find` when exact
  CaDiCaL 3.0.1 is installed; `sat.unsat_proof.verify` when the operator
  installs bundled references and an exactly identified DRAT-trim runtime
- Installed operations: `sat.model.verify` and `sat.lrat.verify` when the
  operator installs the bundled reference checkers

Jacobian always installs canonical CNF, total assignment, and raw DRAT proof
artifact contracts. It exposes `sat.cnf.materialize` to turn bounded
named-variable clauses into that canonical CNF artifact. It conditionally
exposes two exploration capabilities when the pinned CaDiCaL runtime is
available, but does not install the solver.
These artifacts begin as typed, unverified evidence. Storing an assignment does
not establish SAT, and storing proof bytes does not establish UNSAT. An
operator may separately authorize the bundled assignment checker and expose
`sat.model.verify`. Proof verification additionally requires a pinned
DRAT-trim executable with operator-supplied provenance.

`sat.lrat.verify` is deliberately separate from the DRAT operation. Its
bounded v1 accepts only ASCII LRAT additions with ordered positive RUP hints.
RAT hints, deletions, and other LRAT dialect extensions are unsupported and
rejected. It binds the exact CNF object, variable map, DIMACS projection, proof
bytes, limits, checker, and certificate. Malformed, truncated, timed-out,
cancelled, or rejected proofs return `UNKNOWN` and never provide evidence that
the formula is satisfiable.

## Registered descriptors

`JacobianRuntime.core.sat.installation` exposes the content-addressed descriptor
URIs registered by the current runtime:

| Descriptor | Registered name and version | Purpose |
| --- | --- | --- |
| Semantics | `jacobian.sat@1` | Shared propositional-CNF meaning and evidence boundary |
| Schema | `jacobian.canonical-cnf@1` | Canonical named-variable CNF and DIMACS binding |
| Schema | `jacobian.sat-assignment@1` | Total assignment candidate bound to one CNF |
| Schema | `jacobian.sat-proof@1` | Preserved raw DRAT bytes bound to one CNF |
| Schema | `jacobian.sat-lrat-proof@1` | Preserved ASCII LRAT RUP-profile bytes and replay limits |
| Schema | `jacobian.witness-envelope@1` | Exact assignment replay evidence |
| Schema | `jacobian.certificate-envelope@1` | Exact UNSAT proof replay evidence |

The schema URIs are content addressed. They are not capability IDs. The
assignment verification capability appears in `capability://catalog` only
when its checker is operator authorized. The proof verification capability
also requires an available authorized DRAT-trim runtime.

The SAT schemas are model backed. JSON Schema checks their closed structural
shape, and the SAT producer applies the domain cross-field invariants before
publishing a typed artifact. Runtime construction re-registers those model
contracts after restart.

## Canonical CNF

`canonicalize_cnf` accepts variable names plus signed integer clauses. Literal
IDs refer to the caller's variable-name order. It then:

1. validates the bounded ASCII variable names;
2. sorts names and assigns contiguous DIMACS IDs starting at one;
3. renumbers every literal through that map;
4. removes duplicate literals and clauses;
5. omits tautological clauses;
6. sorts literals by variable ID and polarity and sorts the resulting clauses;
   and
7. computes the variable-map and deterministic-DIMACS digests.

An empty clause is retained because it is mathematically material. An empty
formula and unused declared variables are also representable. A literal that
is zero or refers outside the declared variable map is rejected.

The deterministic projection is:

```text
p cnf <variable-count> <clause-count>
<signed literals> 0
```

It uses ASCII, one LF-terminated row per clause, and projection version
`jacobian.dimacs.cnf/v1`. The payload records:

- `variable_map_digest`, over the exact ordered symbolic-name map; and
- `dimacs_digest`, over the exact projected bytes.

Consequently, reordering equivalent source input before canonicalization gives
one artifact identity. Presenting a stored payload with reordered canonical
clauses is invalid rather than a second representation.

### Model-facing CNF materialization

`sat.cnf.materialize` accepts `variable_names` in the caller's literal-ID order
and `clauses` as signed integer arrays. Its closed request model validates all
cross-field constraints and the two-million-literal aggregate bound before the
artifact service writes anything. It returns the canonical `cnf_uri`, schema
and semantics URIs, counts, projection identity, and all exact binding digests.
For at most 4,096 variables it also returns the complete canonical
DIMACS-ID-to-name binding inline. `caller_order_changed` and
`variable_order_note` make lexicographic remapping explicit; model-facing code
should interpret assignments by name rather than assuming that canonical IDs
retain caller order. Larger maps remain available in the exact CNF artifact.

The result has `COMPUTED` assurance and makes no SAT or UNSAT conclusion.
`sat.model.find` and `sat.unsat_proof.find` consume the returned `cnf_uri`;
their evidence can subsequently be passed to the corresponding independent
verification capability when that checker is installed.

## Exact CNF binding

Every assignment and proof contains a `SatCnfBinding` with:

| Field | Bound material |
| --- | --- |
| `cnf_artifact_uri` | Exact stored manifest and lineage identity |
| `cnf_object_digest` | Schema-, semantics-, canonicalizer-, and payload-bound object |
| `cnf_payload_digest` | Canonical CNF payload bytes |
| `variable_map_digest` | Ordered symbolic-name to DIMACS-ID map |
| `dimacs_digest` | Exact solver-facing projection |
| `projection_format`, `projection_version` | DIMACS interpretation |
| `variable_count`, `clause_count` | Declared full-instance scope |

`SatArtifactService` derives this record from a stored canonical CNF. It does
not accept a caller-supplied replacement binding, and it records the CNF
artifact as the evidence artifact's parent.

## Assignment artifacts

An assignment is a strict Boolean vector in variable-map order. Its length must
equal `variable_count`; partial assignments are not part of version 1. It also
records:

- declared scope `FULL_CNF`;
- an available `CapabilityProviderRuntime`, including provider version and
  exact runtime digest; and
- the search resource budget, with a required wall-clock bound and optional
  memory and conflict bounds.

The assignment schema has no conclusion, verification status, checker ID, or
certificate claim.

## CaDiCaL exploration

The base runtime probes `cadical` on `PATH`. It installs `sat.model.find` and
`sat.unsat_proof.find` only when `cadical --version` reports exactly `3.0.1`.
The runtime record uses install tier T2, license identifier `MIT`, the resolved
executable path, platform, supported projection and proof formats, and the
SHA-256 digest of the executable. Every invocation checks that digest before
and after execution. A missing executable, another version, or a changed
executable leaves the capabilities absent or makes the invocation fail without
evidence.

The implementation is tested against upstream tag `rel-3.0.1`, commit
`c60730422e758ef1cebe7aeddf2dda31c996bf04`. Jacobian does not download or
vendor a binary. An operator may build that revision using the upstream
`./configure && make` path and place the resulting `cadical` executable on
`PATH`; the locally built executable receives its own recorded digest.

Both operations accept:

- one exact canonical `cnf_uri`; and
- an enforced wall-time bound of at most 150 seconds plus an optional CaDiCaL
  conflict bound.

They do not accept a memory bound because this adapter does not yet enforce
one. The exact canonical DIMACS bytes are written to an isolated temporary
directory. CaDiCaL runs in a bounded process group with fixed `C` locale,
bounded stdout and stderr, and descendant termination on timeout or excess
output. Remote MCP cancellation is propagated into the bounded process group;
cancelled work retains no solver evidence.

`sat.model.find` accepts only the documented competition protocol: exit 10
plus `s SATISFIABLE` and a unique, range-checked, zero-terminated literal for
every declared variable. It then stores the Boolean vector through
`SatArtifactService.put_assignment`. The result reports the durable
`assignment_uri`, an inline name-to-Boolean map keyed by canonical variable
names,
`ASSIGNMENT_PRODUCED`, `solver_status: SATISFIABLE`, and
`conclusion: UNKNOWN`. The inline map makes the constructed object immediately
inspectable; it does not add assurance. The candidate becomes mathematically
assured only if a later `sat.model.verify` invocation independently accepts it.

`sat.unsat_proof.find` invokes CaDiCaL with `--no-binary` and an explicit proof
path. Exit 20 plus `s UNSATISFIABLE` permits the adapter to read a bounded raw
proof from a non-symlink regular file. The producer deterministically removes
DRAT deletion lines before storing at most 6,000,000 normalized bytes as
`drat-text/v1`. This aligns harmless deletion warnings with the authorized
checker's strict profile without weakening the independent checker:
DRAT-trim still verifies the exact stored addition-only proof against the
exact bound CNF. A proof whose later RAT steps require prior deletions may
therefore be rejected after normalization, but it cannot be promoted without
successful replay. The raw capture limit is separately bounded at 64,000,000
bytes and enforced as an operating-system worker file-size limit on supported
POSIX hosts; exceeding either limit returns a distinct fail-closed diagnostic.
Empty proof bytes are allowed because an input containing an empty clause can
require no added proof step. The result still reports `conclusion: UNKNOWN`;
no solver status or stored bytes establish UNSAT.

Exit 0 is recorded only as solver status `UNKNOWN`. A SAT report from the
proof producer, an UNSAT report from the model producer, or any bounded attempt
without the requested evidence returns `NO_*_PRODUCED` and `UNKNOWN`. Timeout,
non-protocol exit, inconsistent text status, malformed or partial model,
oversized output, unsafe proof file, and runtime replacement are operational
failures. None creates solver evidence or an opposite conclusion.

## Assignment verification

`sat.model.verify` accepts one `assignment_uri`. Before
starting a checker process, its adapter:

1. validates the stored assignment with the model-backed schema;
2. resolves the CNF named inside the assignment;
3. derives the current CNF binding from that stored artifact;
4. requires every binding field to match and the CNF to be an assignment
   parent; and
5. materializes a `sat.assignment@1` witness bound to the exact CNF,
   assignment, and SAT semantics.

The authorized checker then runs in a clean process. Its implementation uses
only the Python standard library and no solver code or Jacobian SAT contract
implementation. It independently validates the closed canonical CNF shape,
variable ordering, clause ordering, payload, variable-map and DIMACS digests,
assignment binding, total strict-Boolean vector, evidence bindings, and
lineage. It returns `TRUE` only after evaluating every clause successfully.

Acceptance creates the ordinary runtime `VerificationRecord` and allows the
capability result to report `VERIFIED`. Assignment rejection reports
`UNKNOWN`: it does not establish UNSAT. A malformed or misbound artifact fails
before checker dispatch. Timeout, checker error, cancellation, and incomplete
execution likewise remain non-verified and carry no SAT or UNSAT conclusion.
Direct witness replay makes no enumeration-completeness claim.

## Raw proof artifacts

A proof artifact preserves the exact raw bytes as canonical base64 and records
their SHA-256 digest. Version 1 binds:

- format `DRAT`;
- format version `drat-text/v1`;
- encoding `BASE64`;
- declared scope `FULL_CNF`;
- the complete CNF binding;
- the exact producer runtime; and
- the producing search budget.

The artifact layer does not parse the proof or infer UNSAT. Malformed,
truncated, or adversarial bytes may be retained as unverified evidence for
later fail-closed replay. The current artifact store has a 10 MiB payload
limit, and this contract bounds the base64 field to 8,000,000 characters.

## UNSAT proof verification

`sat.unsat_proof.verify` accepts one `proof_uri`. Its adapter
resolves the raw proof and exact parent CNF, re-derives every CNF binding field,
requires the source lineage, and materializes a `sat.unsat-proof@1`
`CertificateEnvelope`. The certificate binds the CNF claim, proof candidate,
SAT semantics, exact artifact URIs, payload digests, and parents.

Its model-facing output always reports
`verified_claim_scope: "CANONICAL_CNF_ONLY"`. Acceptance establishes only that
the exact bound canonical CNF is unsatisfiable. It does not establish that a
caller-authored CNF correctly encodes a graph property, threshold, coloring
instance, or other domain claim. Such a claim needs a domain-owned result and
an operator-authorized domain checker that independently binds or reconstructs
the encoding.

The capability is installed only when bundled references are enabled and the
operator authorizes an available DRAT-trim runtime. The supported runtime is
upstream release `v05.22.2023`, source commit
`2e5e29cb0019d5cfd547d4208dca1b3ec290349f`. DRAT-trim does not expose a
stable machine-readable version identity, so Jacobian requires a sibling
`drat-trim.jacobian-runtime.json` file with exactly:

```json
{
  "runtime_manifest_version": "1",
  "provider": "drat-trim",
  "release_tag": "v05.22.2023",
  "source_repository": "https://github.com/marijnheule/drat-trim",
  "source_commit": "2e5e29cb0019d5cfd547d4208dca1b3ec290349f",
  "executable_sha256": "sha256:<64 lowercase hexadecimal digits>"
}
```

The sidecar is an operator assertion about the installed build, not upstream
attestation. The executable digest and provenance become part of the
authorized checker identity. The registry rehashes the executable when the
checker is selected; the clean worker checks it before and after replay and
binds it into the verification environment digest.

The standard-library-only checker adapter independently validates the closed
CNF, proof, certificate, evidence bindings, payload digests, and lineage. It
reconstructs canonical DIMACS and admits a bounded ASCII `drat-text/v1`
profile. Malformed clauses, duplicate or complementary literals, integer
overflow, deletion of the empty clause, or non-deletion steps after the empty
clause are rejected before external dispatch. Cleanup deletions after the
empty clause are admitted because CaDiCaL may emit them; they cannot alter the
already-derived contradiction. A fixed benign comment is prepended to force
DRAT-trim's text parser; it does not alter the stored proof bytes.

DRAT-trim runs its official forward UNSAT check against temporary CNF and proof
files with `-f -W`, fixed locale, bounded time and output, and the exact
authorized executable. Acceptance requires exit zero, empty stderr,
protocol-only output, and exactly one `s VERIFIED` line. Any other status,
warning, malformed output, excessive output, mutation, cross-CNF replay,
runtime replacement, timeout, cancellation, or crash yields no mathematical
conclusion.

Acceptance creates the ordinary runtime `VerificationRecord`, bound to the
certificate and all three artifacts, and permits `VERIFIED_UNSAT` with
conclusion `TRUE`. Rejection reports `UNKNOWN`; it does not establish SAT.

## Public reproductions

The unscored manifest
[`sat-small`](../../../../benchmarks/datasets/public-reproductions-v1/sat-small/)
replays three public cases through the real installed backends:

- the complete `BOOL-MUS-001` formula, without treating later shrinking as
  part of this capability slice;
- one small satisfiable CNF; and
- the three-pigeons/two-holes UNSAT instance.

The manifest records `scored: false`; these cases exercise compatibility and
regression behavior and are never hidden evaluation. Run the actual
CaDiCaL-to-checker path with:

```sh
uv run pytest -n 0 tests/boundary/providers/external_sat/test_sat_public_reproductions.py
```

The `BOOL-MUS-001` replay exposed CaDiCaL cleanup deletions after its empty
clause. Preserving the raw proof and using DRAT-trim's forward check accepted
that valid producer output without weakening rejection of concatenated
post-contradiction additions.

## Trust boundary

The two producer/checker pairs remain separate:

- CaDiCaL output and raw proof storage remain unverified;
- assignment rejection never establishes UNSAT;
- DRAT-trim rejection never establishes SAT; and
- only the operator-authorized checker accepting the exact bound certificate
  may create a `VERIFIED` record.

CaDiCaL is a producer, not a checker. Its status is retained only as an
unverified operational report. DRAT-trim replay is an independent
clean-process capability and authorization boundary.
