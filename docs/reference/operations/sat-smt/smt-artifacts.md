# SMT Alethe artifact contracts

[Documentation home](../../../index.md)

- Status: Experimental producer and verifier contracts
- Producer operation: `smt.unsat_proof.find`, backed by the exact packaged
  cvc5 1.3.4 Python distribution
- Verification operation: `smt.unsat_proof.verify` for the pinned zero-hole
  `QF_UF` compatibility profile when an operator-authorized Carcara runtime is
  installed

Jacobian's first SMT slice preserves one exact quantifier-free SMT-LIB query
and the raw Alethe bytes emitted by cvc5. It does not expose a broad
`smt.solve` workflow. A cvc5 `unsat` report, stored proof bytes, or the absence
of lexical `hole` markers is computed evidence, not independent verification.
Every producer result therefore carries `conclusion: UNKNOWN`.

The base package and locked development environment both include this exact
provider. Carcara remains separately operator-installed for independent replay.

## Registered descriptors

`JacobianRuntime.core.smt.installation` exposes the content-addressed descriptor
URIs registered by the current runtime:

| Descriptor | Registered name and version | Purpose |
| --- | --- | --- |
| Semantics | `jacobian.smt.qf-unsat@1` | Quantifier-free single-query meaning and evidence boundary |
| Schema | `jacobian.smt-problem@1` | Exact bounded SMT-LIB 2.6 input |
| Schema | `jacobian.smt-alethe-proof@1` | Raw cvc5 Alethe bytes bound to one exact input |

The schemas are model backed. Their closed structural and cross-field
invariants apply before the SMT producer publishes a typed artifact.

## Pinned SMT-LIB profile

Profile `jacobian.smtlib2.qf-unsat/v1` admits exactly one query in one of:

- `QF_UF`;
- `QF_LIA`; or
- `QF_LRA`.

The source is at most 1,000,000 ASCII bytes, uses LF line endings, ends in LF,
and has a maximum parenthesis nesting depth of 512. Its top-level commands are
limited to:

- one leading `set-logic` equal to the separately declared logic;
- `declare-sort`, `declare-fun`, or `declare-const`;
- zero or more `assert` commands; and
- one final argument-free `check-sat`.

Incremental commands, solver-option changes, definitions, assumptions,
result-retrieval commands, reset, include, multiple queries, quantifiers, and
theories outside the selected logic are not part of version 1. The contract
scanner handles comments, strings, and quoted symbols when identifying
top-level command boundaries. The isolated cvc5 parser then independently
rejects source that is not valid in the declared logic.

The problem artifact preserves the exact text and SHA-256 digest. It does not
claim that equivalent presentations have one canonical identity.

## Provider identity and isolation

The catalog entry requires the exact `cvc5==1.3.4` distribution with the
expected parser, solver, proof-component, and proof-format APIs. Its runtime
record uses:

- install tier `T1`;
- license identifier `BSD-3-Clause`;
- digest kind `PYTHON_DISTRIBUTION_RECORD`;
- feature flags `smt-lib-2.6` and `alethe-proof-production`; and
- profile and proof-format configuration.

The digest identifies the installed wheel RECORD manifest; it does not claim
to rehash every package byte. Provider identity and successful execution
remain separate from mathematical assurance.

The adapter does not call the native solver in the MCP server process. It
starts an isolated Python worker in its own bounded process group, fixes locale
and timezone, and applies the declared wall-time both as cvc5's
`tlimit-per` and as a parent process deadline. Worker stdout is a closed JSON
protocol capped at 4 KiB, stderr is capped at 64 KiB, and raw proof capture is
capped at 6,000,000 bytes. Timeout or stream overflow terminates descendants.

## Alethe proof production

`smt.unsat_proof.find` accepts:

```json
{
  "logic": "QF_UF",
  "smtlib_text": "(set-logic QF_UF)\n(assert false)\n(check-sat)\n",
  "resource_budget": {
    "wall_seconds": 5
  }
}
```

The adapter first materializes the exact input problem, then invokes cvc5 with
proof production enabled and serializes the full proof as Alethe. An
`UNSATISFIABLE` solver report is usable only when the worker also creates one
bounded regular proof file and its reported lexical hole count matches the
captured bytes.

The proof artifact binds:

- the problem artifact URI, object digest, payload digest, logic, profile,
  language, and exact SMT-LIB digest;
- format `ALETHE` and version `cvc5.alethe/1.3.4`;
- exact proof bytes as canonical base64 plus their SHA-256 digest;
- the exact cvc5 executable or distribution identity;
- the enforced wall-time budget; and
- `alethe_hole_count` plus `contains_holes`.

Hole metadata is an inspectable routing signal, not proof checking. It counts
the exact byte marker `:rule hole`; it does not establish that every other rule
is supported or valid.

The result reports `PROOF_PRODUCED`, the problem and proof URIs, solver status,
and hole metadata at assurance level `COMPUTED`. `SATISFIABLE` or `UNKNOWN`
returns `NO_PROOF_PRODUCED`; it does not produce a model, prove SAT, or imply
anything from failure to find a proof.

## Strict Carcara verification

`smt.unsat_proof.verify` accepts one `proof_uri`. It resolves
the exact proof and parent problem, re-derives the problem binding, and creates
an `smt.unsat-proof@1` `CertificateEnvelope`. The certificate binds the
problem claim, proof candidate, SMT semantics, exact artifact URIs, payload
digests, and required lineage.

The operation is installed only when bundled references are enabled and the
operator authorizes an available Carcara runtime. The pinned checker is
Carcara `1.1.0` at source commit
`394edbb15ba95c47893f1d821fddde7e016af178`, the revision selected by cvc5
1.3.4's
[`get-carcara-checker`](https://github.com/cvc5/cvc5/blob/cvc5-1.3.4/contrib/get-carcara-checker)
helper. Build that exact revision and place the executable on `PATH`. The
pinned source requires Cargo 1.87 or newer.

```sh
cargo install \
  --git https://github.com/ufmg-smite/carcara.git \
  --rev 394edbb15ba95c47893f1d821fddde7e016af178 \
  --locked carcara-cli
```

Add a sibling `carcara.jacobian-runtime.json` file containing exactly:

```json
{
  "runtime_manifest_version": "1",
  "provider": "carcara",
  "version": "1.1.0",
  "source_repository": "https://github.com/ufmg-smite/carcara",
  "source_commit": "394edbb15ba95c47893f1d821fddde7e016af178",
  "compatible_cvc5_version": "1.3.4",
  "executable_sha256": "sha256:<64 lowercase hexadecimal digits>"
}
```

The sidecar is an operator assertion about the installed build, not upstream
attestation. Jacobian checks the exact version output, required command-line
surface, executable digest, and provenance before authorization. The registry
rehashes the executable when the checker is selected; the clean worker checks
it before and after replay and binds it into the verification environment
digest.

The checker module uses only the Python standard library and does not import
the SMT producer contracts or cvc5. It independently validates the closed
problem, proof, certificate, evidence bindings, payload digests, producer
identity, and lineage. Version 1 admits only:

- logic `QF_UF`;
- profile `jacobian.smtlib2.qf-unsat/v1`;
- cvc5 `1.3.4` Alethe format `cvc5.alethe/1.3.4`; and
- proof bytes whose bound metadata and lexical hole count both report zero.

The external command is fixed to:

```text
carcara check --strict-parsing --parse-hole-args \
  --allow-int-real-subtyping --expand-let-bindings PROOF PROBLEM
```

The checker never passes `--ignore-unknown-rules` or `--allowed-rules`;
Carcara documents both paths as treating unsupported rules as holes.
Acceptance requires exit zero, exact stdout `valid` plus one LF, empty stderr,
bounded output and time, and an unchanged runtime digest. In particular,
`holey` is rejected even though Carcara returns exit zero for that status.

Acceptance creates the ordinary runtime `VerificationRecord` and permits
`VERIFIED_UNSAT` with conclusion `TRUE`. Rejection reports `UNKNOWN`; it does
not establish satisfiability. Carcara is an independently implemented
proof checker, not a formally verified checker, so this profile narrows the
trusted implementation boundary without eliminating it.

## Compatibility matrix and reproduction cases

The pinned public spike exercises three small cases:

| Logic | Query shape | cvc5 1.3.4 proof | Strict Carcara result | Jacobian assurance |
| --- | --- | --- | --- | --- |
| `QF_UF` | `a = b` and `not (a = b)` | zero holes | `valid` | `VERIFIED` |
| `QF_LIA` | integer `x >= 1` and `x <= 0` | one hole | `holey` | `COMPUTED`, `UNKNOWN` |
| `QF_LRA` | real `x > 1` and `x < 0` | multiple holes | `holey` | `COMPUTED`, `UNKNOWN` |

These observations are version-bound regression cases, not a compatibility
claim for all `QF_UF` inputs. The checker contract intentionally excludes
`QF_LIA` and `QF_LRA` even if a future producer happens to emit a zero-hole
proof; expanding the intersection requires new compatibility and mutation
evidence. cvc5's own
[Alethe output documentation](https://cvc5.github.io/docs/latest/proofs/output_alethe.html)
also shows that proof output may contain untranslated rewrites represented as
holes.

## Fail-closed boundary

No proof artifact is retained when:

- source validation or the cvc5 parser rejects the query;
- the worker times out, crashes, or exceeds an output limit;
- its JSON protocol, status, proof-presence flag, or hole count is malformed;
- a non-UNSAT status carries proof material;
- an UNSAT status lacks one bounded regular proof file; or
- the proof bytes disagree with the worker's hole metadata.

Operational failure returns `ERROR` or `TIMEOUT`, heuristic assurance, and no
mathematical conclusion. The already materialized exact problem may remain as
the operation's sole artifact.

The verifier additionally rejects holes, unsupported logics, unknown-rule
mutations, cross-problem replay, malformed or extra checker output, warnings,
runtime replacement, timeout, cancellation, and crash. None of these failures
is converted into SAT or UNSAT.
