# Atomic capability portfolio

[Documentation home](../index.md)

- Status: Maintainer research record; all proposed capability IDs are
  provisional
- Research snapshot: 2026-07-26
- Repository baseline: `dc1fa846dd1c`
- Scope: Formal-first ordering of external mathematical backends and the
  smallest useful capability slices to expose

## Decision

Keep the five-tool MCP surface small. External systems should appear as
versioned capability adapters behind `math.find` and
`math.run`, not as one new MCP tool per backend.

Build in this order:

1. measure optional-backend availability and provenance consistently;
2. run a small Lean declaration-discovery experiment against the already
   pinned Lean environment;
3. implement an independently replayable SAT model and UNSAT-proof lane;
4. add a theory-bounded cvc5 Alethe proof lane;
5. add exact shared mathematics through Python-FLINT, followed by typed SymPy
   expression operations;
6. expand into first-order proving and narrowly justified domain branches; and
7. add another large proof-assistant or SageMath environment only when
   workflow evidence shows a gap that the lighter providers cannot cover.

This is formal-first in the verification sense: operations with a clear
certificate and independent replay path precede broad computation whose
provider merely reports success. It does not mean installing every proof
assistant before useful exact arithmetic.

The first new implementation target should be the SAT vertical slice. The Lean
discovery slice comes first as a contract and evaluation spike because its
backend is already present, but the existing Lean REPL pilot has not yet shown
enough agent benefit to make additional Lean operations default capabilities.

Do not add `lean.execute`, `sympy.eval`, `sagemath.eval`, `solver.solve`, or
another arbitrary code or command surface. Capability IDs describe one
mathematical outcome; the provider remains descriptor and provenance data.

## Research method and evidence boundary

This plan follows the repository's capability-discovery process. It mines open
workflows with their answers visible, then hands concrete proposals to held-out
comparative evaluation. It is not itself evidence that a capability improves
autonomous performance.

The source bundle is:

| Source | Snapshot and use |
| --- | --- |
| Current Jacobian catalog, descriptors, source, and tests | Baseline at `dc1fa846dd1c`; establishes existing operations and trust rules |
| [Mathematical scenarios](../reference/scenarios/math-scenarios.md) and [workflow evaluation plan](../reference/evaluations/benchmark-contracts.md) | Repeated moves, failure patterns, and public reproduction cases |
| Maintainer documentation and repositories linked below | Backend contracts, certificate formats, supported domains, installation paths, and licenses |
| [PyPI JSON metadata][pypi-json] | Dated compressed-wheel size snapshot; not installed size or runtime memory |
| Local pinned Lean installation | One development-host measurement used only to show ecosystem scale |

Public examples and known answers are suitable for contract reproduction and
regression. They must not become hidden evaluation cases. Comparative claims
require frozen, held-out variants and independent oracles under the
[agent evaluation protocol](../reference/evaluations/evaluation-methods.md).

Package licenses in this document are discovery inputs, not a completed legal
review. Before vendoring, redistributing, or enabling a backend by default,
record the exact version, transitive native components, applicable license
files, and redistribution decision.

## Current baseline and observed gaps

The source runtime now combines the original artifact, claim, transformation,
evaluation, search, witness, certificate, shrinking, experiment,
parameter-region, and memory operations with an explicit
[domain operation library](../reference/domain-operation-library.md). Its
built-in bundles cover arithmetic, number theory, combinatorics, finite sets,
sequences, geometry, graph optimization and invariants, matrices, lattices,
polynomials, validated analysis, finite probability, and rational
optimization. Lean capabilities also expose declaration discovery, premise
retrieval, typed proof states, proof-edit validation, and independent replay.

The installed count is intentionally not frozen here. Provider availability,
configured exclusions, bundled references, and operator-authorized checkers
change catalog membership; `capability://catalog` is the authoritative
inventory. Backend names such as `reference.solve` remain intentionally absent.

The broad portfolio closes several discovery gaps recorded in the original
baseline, but it does not remove the evaluation gates in this document.
Remaining work is evidence-led: improve discovery and composition over the
larger catalog, add independent checkers where exact relations justify them,
and extend domains only when repeated workflows demonstrate a missing atomic
outcome.

The existing graph pilot also gives a warning for every new adapter. Returning
only an artifact URI made a correct isomorphic graph impossible for the agent
to bind to the original artifact. Small useful outputs therefore need an inline
typed summary as well as a durable artifact and explicit relationships.

## Selection rubric

Score a proposed capability from one to five on each factor. The weighted score
orders experiments; it does not override a hard trust or operability gate.

| Factor | Weight | Question |
| --- | ---: | --- |
| Independent evidence path | 25% | Can a separate checker replay the exact claim, scope, candidate, and certificate? |
| Cross-domain leverage | 20% | Will several workflows reuse the mathematical outcome? |
| Repeated workflow demand | 15% | Does current scenario, transcript, dataset, or user evidence contain this move? |
| Installation and operations | 15% | Is installation pinned, bounded, portable, and reasonably small? |
| Atomic contract quality | 10% | Is there one typed, agent-visible outcome without arbitrary execution? |
| Maintenance and licensing | 10% | Is the provider maintained and practical to redistribute or install? |
| Current fit and sunk cost | 5% | Does it reuse installed infrastructure, schemas, or existing scenarios? |

A proposal is not implementation-ready if any of these hard gates fail:

- no stable typed input and output schema;
- no bounded subprocess, timeout, cancellation, and output-size policy;
- no reproducible version pin or availability check;
- no truthful mapping from backend outcomes to execution, completeness,
  evidence type, and assurance;
- no public micro-case plus adversarial mutation;
- a claimed `VERIFIED` path without an operator-authorized independent
  checker; or
- a broad command language where a smaller mathematical contract will do.

Failure to find an object, a timeout, an unsupported theory, an incomplete
enumeration, or a checker error remains `UNKNOWN` or an execution failure. None
of these is a mathematical conclusion.

## Installation tiers and size snapshot

Keep the base environment small and advertise an adapter only when its pinned
provider is available.

| Tier | Deployment rule | Intended use |
| --- | --- | --- |
| T0 | No new external runtime beyond Jacobian's locked base | Pure checkers, schemas, and existing Z3 use |
| T1 | Optional `uv` extra with maintained wheels | Python-FLINT, SymPy, cvc5 Python bindings |
| T2 | Pinned external executable with checksum and explicit operator enablement | CaDiCaL, DRAT-trim, Carcara, nauty, E, Vampire |
| T3 | Isolated container, remote worker, or separately managed ecosystem | Lean plus Mathlib, Rocq platform, Isabelle, SageMath |

For rough comparisons, `S` means at most 15 MiB compressed, `M` means more
than 15 through 75 MiB, `L` means more than 75 through 500 MiB, and `XL` means
more than 500 MiB or an ecosystem whose build and data cost dominates its
download artifact. These are planning bands, not release limits.

The largest wheel in each package's PyPI JSON response on 2026-07-26 was:

| Package | Version | Largest compressed wheel |
| --- | --- | ---: |
| SymPy | 1.14.0 | 6.0 MiB |
| Python-FLINT | 0.9.0 | 9.9 MiB |
| cvc5 | 1.3.4 | 13.1 MiB |
| z3-solver | 5.0.0.0 | 39.2 MiB |
| highspy | 1.15.1 | 6.9 MiB |
| cypari2 | 2.2.4 | 9.1 MiB |

These are upstream snapshot versions, not selected Jacobian pins. Compressed
wheel size omits caches, shared environments, proof libraries, solver data,
and peak memory. As a counterexample, the pinned Lean 4.31.0 toolchain occupies
2.8 GiB on the development host before treating Mathlib as a separate
operational concern. The [Lean installation guide][lean-install] also calls out
the extended first Mathlib download and cache retrieval.

Every implementation issue must replace these discovery estimates with a
reproducible cold-install measurement, installed disk use, cold start, peak
memory on its reproduction cases, and platform coverage.

## Ranked provider portfolio

The rank is the default investigation and implementation queue. A lower-ranked
system can move earlier when a concrete workflow repeatedly needs its outcome.

| Rank | Provider | Smallest useful outcomes | Install | Verification boundary | Decision |
| ---: | --- | --- | --- | --- | --- |
| 0 | Existing Z3 | Internal bounded search and differential checking | T0, already locked | Never treat Z3's own status as independent verification | Retain as a provider; do not add a Z3-branded self-verifying capability |
| 1 | Lean 4, Mathlib, and a thin maintained interaction layer | Declaration search and inspection; completed source still goes to `lean.check` | T3/XL, but already pinned for references | Existing independent Lean replay for completed source; retrieval itself is unverified | Contract spike and paired evaluation, not a default expansion yet |
| 2 | CaDiCaL plus DRAT-trim, with the pinned Lean checker as the authority candidate | Find a SAT assignment; emit an UNSAT proof; check each independently | T2/S-to-M native build | Direct assignment replay for SAT; separate proof checker for UNSAT | First new formal vertical slice |
| 3 | cvc5 plus Carcara | Produce and check Alethe UNSAT proofs in explicitly supported theories | T1/S plus T2 Rust checker | Carcara or later proof reconstruction, bound to exact SMT-LIB and theory profile | Implement after SAT; unsupported proof rules remain unverified |
| 4 | Python-FLINT and Arb | Exact rational linear algebra, integer matrices, polynomials, and rigorous ball enclosures | T1/S | Small independent exact checkers where available; Arb output remains computed until independently replayed | First shared exact-mathematics provider |
| 5 | SymPy | Typed expression normalization and symbolic transformations | T1/S | Verify only a separately checkable relation; never trust simplifier success alone | Add after FLINT with a non-string typed AST |
| 6 | E or Vampire with TPTP/TSTP | First-order model or derivation production | T2/S-to-M | Independent TSTP step checking or proof reconstruction; coverage must be measured | Formal research slice after SMT |
| 7 | Metamath | Replay a proof against a pinned database | T2/S | Small independent verifier bound to exact database revision | Low-cost second-kernel pilot, but narrow corpus leverage |
| 8 | nauty and Traces | Canonical label, isomorphism witness, nonisomorphic generation | T2/S | Replay permutations directly; exhaustive-generation claims need checked scope | First graph-specific expansion when Graph Atlas becomes limiting |
| 9 | PARI/GP through cypari2 | Number-field and arithmetic invariants not already covered by FLINT | T1/S when wheels apply | Result-specific certificates or unverified computation | First number-theory branch, only for a concrete missing outcome |
| 10 | Normaliz and PyNormaliz | Cone, lattice-point, Hilbert-basis, and Ehrhart computations | T2/T3 | Membership is easy to replay; minimality and completeness need stronger certificates | Polyhedral branch after the exact core |
| 11 | GAP | Finite-group and discrete-algebra computations | T2/T3, package set varies | Domain-specific checker or unverified exact computation | Add per group-theory workflow, not as a GAP command shell |
| 12 | Singular | Gröbner and standard bases over declared coefficient domains | T2/M | Independently reduce generators and S-polynomials for a checked basis claim | Strong algebra branch after shared polynomial schemas |
| 13 | HiGHS, later paired with an exact certificate checker | LP/MIP/QP candidate solutions and bounds | T1/S | Exact primal/dual/Farkas or VIPR-style replay; floating status is not proof | Explore first; verification only with exact certificates |
| 14 | fplll and fpylll | LLL reduction and lattice witnesses | T2/M | Check the exact transformation, determinant or unimodularity, and reducedness conditions | Later lattice branch |
| 15 | Rocq | Check a pinned Rocq proof and later inspect declarations | T3/XL ecosystem | Rocq kernel under an operator-authorized installation | Defer until a corpus or user workflow needs a second large assistant |
| 16 | Agda | Check a pinned Agda development | T2/T3; prebuilt compiler, libraries add cost | Agda type checking bound to exact libraries | Defer; lower current math-corpus leverage |
| 17 | Isabelle | Replay an Isabelle session | T3/XL bundle | Isabelle session build under a pinned installation | Defer despite strong formal coverage because deployment is heavy |
| 18 | SageMath | Only concrete operations missing from lighter providers | T3/XL environment | Result-specific external checker; Sage success is never verification | Optional container or remote fallback, not an early dependency |
| 19 | Macaulay2 | Specialized commutative algebra and algebraic geometry | T3 | Result-specific replay or unverified computation | Defer until Singular and Normaliz leave a measured gap |

Rocq, Agda, and Isabelle are low in the default queue because they duplicate
the expensive proof-assistant environment role before Jacobian has measured a
corpus need. This is not a judgment about their mathematical quality.

SageMath is deliberately late. Its value is breadth, but that breadth overlaps
FLINT, PARI, GAP, Singular, and many other systems while imposing an
environment-scale install. Three independently observed, high-value operations
that cannot be served cleanly by lighter providers should be the minimum
trigger for a Sage integration spike.

## Workflow ledger

The ledger separates observed process evidence from attractive but unsupported
backend ideas.

| Evidence | Mathematical move | Current support or gap | Proposed change | Backend | Verification boundary | Public reproduction | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lean workflow and pinned REPL pilot | Find a relevant declaration, inspect its type, then construct source | `lean.check` replays completed source; discovery and proof-state handoff are missing, and the REPL cannot yet return replayable source | `lean.declaration.search` and `lean.declaration.inspect` experiment | Lean environment; spike maintained Lean REPL or Pantograph only where needed | Search is unverified; final source uses existing `lean.check` | Search and inspect known Mathlib declarations, then replay a small proof | Ready for contract spike and ablation, not default exposure |
| `BOOL-MUS-001`, finite quantified cases, and repeated solver-backed construction | Find a Boolean model or produce evidence of UNSAT | No shared SAT artifact and certificate lane | Split model finding/checking from UNSAT proof production/checking | CaDiCaL and DRAT-trim, with pinned Lean authority evaluation | Direct model evaluator and separate proof checker | Small satisfiable CNF, pigeonhole UNSAT, corrupted assignment, truncated and misbound proof | Ready for implementation |
| Existing Z3 use and cross-domain SMT demand | Establish UNSAT in EUF or linear arithmetic | Z3 is a search dependency, not an independent verifier | Theory-bounded Alethe proof capabilities | cvc5 and Carcara | Separate checker; reject unsupported rules and theories without conclusion | Official cvc5 Alethe examples plus mutated premises and theory-profile mismatch | Ready after SAT |
| Matrix, polytope, and exact-algebra scenarios | Construct an exact rational solution and expose replayable algebra | Current exact work is scenario-specific | Exact rational solution first; Hermite form and factorization later | Python-FLINT | Independent rational arithmetic and relation checks | Rational systems with a valid solution, malformed denominator, wrong transform, and inconsistent-system non-conclusion | Ready after proof lanes |
| Symbolic reasoning in general workflows | Normalize or transform a declared expression | No shared typed expression schema; arbitrary parsing would be unsafe | Typed AST and narrowly named transformations | SymPy, with FLINT for polynomial canonical forms | Verify only canonical coefficients or another explicit relation | Polynomial normalization and deliberate assumption or branch-cut traps | Contract research, then implementation |
| Graph Atlas pilot and larger graph workflows | Canonicalize, test isomorphism, or enumerate beyond order seven | Atlas is bounded and small; no observed held-out need yet | Canonical label and isomorphism witness before enumeration | nauty/Traces | Replay the permutation; bind any completeness claim to exact order and generator | Reproduce Atlas canonical classes and mutated permutation | Research queue |
| `POLY-SEP-001` and future lattice-point cases | Compute cone or integer-semigroup data | Exact separation exists; Hilbert bases and Ehrhart data do not | Add one Normaliz outcome when a scenario needs it | Normaliz/PyNormaliz | Separate membership from minimality and completeness obligations | Tiny cones with independently enumerated bounded sections | Research queue |
| Tiny optimization scenario | Find an LP/MIP candidate and bound | Solver results cannot currently be promoted safely | Candidate solution first, exact certificate checking second | HiGHS and a maintained exact checker where formats align | Exact primal/dual/Farkas or integer proof replay | Tiny rational LP with perturbed objective, infeasible claim, and timeout | Research queue |

The PARI, GAP, Singular, fplll, Rocq, Agda, Isabelle, SageMath, and Macaulay2
rows in the ranked portfolio are backend hypotheses. They need at least one
workflow ledger row with repeated evidence before implementation.

## Wave 0: optional-provider groundwork

Complete this once, before adding several backends:

1. Define adapter availability metadata: provider name and version, executable
   or package digest, supported platform, install tier, license identifier and
   file, detected feature flags, and checker identity where applicable.
2. Advertise only installed, healthy adapters. An absent optional backend is a
   catalog condition, not an invocation-time surprise.
3. Put provider choice in descriptors and provenance, not in operation IDs.
   Register at most one default provider for an ID; compare alternative
   providers in internal differential tests or explicitly versioned
   experiments.
4. Standardize subprocess limits, sanitized environments, temporary
   directories, input and output byte limits, timeout, cancellation, exit-code
   mapping, and log redaction.
5. Add a repeatable backend-measurement command that records cold install,
   installed disk, cold start, peak resident memory, and reproduction-case
   runtime.
6. Keep T2 and T3 adapters operator-enabled. Do not expand the base dependency
   set merely so the catalog looks broad.

This groundwork changes packaging and descriptors, not the mathematical
assurance model and not the top-level MCP tool count.

The initial availability, identity, install-tier, and measurement contract is
implemented in the
[provider runtime reference](../reference/provider-runtime.md). It covers
source-tree, Python RECORD, and executable identities; fail-closed catalog
registration; compact result provenance; and the repeatable
`provider-measure` command. Provider-specific subprocess hardening and
reproduction measurements remain part of each later backend slice.

## Wave 1: Lean discovery experiment

Treat these names as contract sketches:

- `lean.declaration.search` consumes a pinned environment, a structured name or
  type-pattern query, namespace filters, and a result budget. It returns
  declaration names, types, source locations when available, and match reasons.
- `lean.declaration.inspect` consumes one exact declaration name and returns
  its elaborated type, kind, namespace, documentation and source metadata when
  available, and the exact environment digest.

Both are `EXPLORE` retrieval operations. Neither verifies a theorem merely by
finding its declaration. A completed source artifact must still pass
`lean.check`.

Do not expose tactic stepping, `goal.decompose`, or `premise.apply` by default
in this wave. The current REPL spike can expose goal state but cannot turn a
completed tactic state into the originating command or a replayable proof
artifact. Add those operations only if held-out transcripts show that
declaration retrieval is insufficient and a maintained interaction layer
preserves a clean source handoff.

The public reproduction should retrieve and inspect known Mathlib declarations,
then construct a small proof checked by `lean.check`. The paired evaluation
must use held-out declaration families. Keep the new capabilities only if they
improve task completion or provenance enough to justify their catalog and token
cost; otherwise improve `lean.check` examples or use an agent-side retrieval
skill.

The contract spike implemented on 2026-07-26 chose an exact-constant type
pattern over pretty-printed type text. All named constants must occur in Lean's
elaborated type expression. This keeps the query structural and makes a full
no-match Mathlib scan practical without a custom index. On the development
host with warm filesystem caches, direct helper measurements were:

- 11.1 seconds for a one-result Mathlib name search;
- 29.8 seconds to exhaust 626,944 public declarations for a no-match type
  pattern; and
- 69.0 seconds for the tested public search, inspect, and independent
  `lean.check` composition.

The first two operations return `COMPUTED` retrieval evidence and bind the
exact pinned environment-manifest digest. The last operation alone returned
`VERIFIED`. These measurements established a workable spike, not portfolio
lift.

The held-out paired pilot completed on 2026-07-26. Both `List.revzip`
treatments used discovery and reached independently verified proofs, but the
controls also passed. Discovery added 226--288 seconds, 208k--365k input
tokens, and three to six additional tool execution errors in those pairs. A
second statement family was solved without selecting discovery. The decision
is to revise rather than stabilize, consolidate, or retire:

- retain search and exact inspection as separate experimental atomic outcomes;
- do not recommend them or add goal-stepping capabilities yet;
- replace repeated Mathlib startup and full scans with a reusable pinned index
  or persistent query service instead of merely raising the timeout;
- compact catalog discovery; and
- rerun the frozen held-out evaluation before changing recommendation status.

See [the task and verifier validation boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation)
for scorer invariants, per-run measurements, and the excluded operationally
invalid pair.

The indexed follow-up is implemented. It derives a compact catalog from the
pinned environment's imported module metadata, binds it to the exact
environment digest, rechecks its byte identity before and after reuse, and
keeps final candidate and type matching inside Lean. A fresh held-out
`List.revzip` search completed in 28.886 seconds instead of timing out; catalog
reuse completed in 9.568 seconds and direct exact inspection in 9.146 seconds.
Fresh and reused outputs agreed on declarations, stop reason, and exact scan
count.

The frozen autonomous rerun passed in both conditions with no tool errors, but
neither treatment selected discovery. That run therefore does not measure
intervention lift. Search and inspection remain experimental and
non-recommended. Before changing recommendation status or exposing interactive
goal tools, add harder held-out statements where direct automation does not
already solve the proposition.

## Wave 2: SAT certificate vertical slice

Use four outcomes rather than one ambiguous solver call:

- `sat.model.find` attempts to construct one assignment for a canonical CNF
  artifact;
- `sat.model.verify` checks that an assignment satisfies every bound clause;
- `sat.unsat_proof.find` attempts to produce a proof for that exact CNF; and
- `sat.unsat_proof.verify` checks the proof in a separate process.

`sat.model.find` failing to find a model does not establish UNSAT.
`sat.unsat_proof.find` failing to produce a proof does not establish SAT.

Start with a domain-owned canonical CNF artifact and a deterministic DIMACS
projection. Bind every assignment or proof to the source artifact digest,
variable map, projection version, declared scope, provider version, and
resource budget. Preserve the raw proof as a durable artifact but keep bounded
inline summaries small.

The artifact-contract checkpoint was implemented on 2026-07-26. The base
runtime now registers model-backed schemas for canonical CNF, total assignment,
and raw DRAT proof artifacts without installing a solver, checker, or SAT
capability. Canonicalization deterministically renumbers the sorted variable
map, removes duplicate literals and clauses, omits tautologies, orders the
remaining clauses, and binds the resulting DIMACS bytes. Assignment and proof
artifacts bind the exact CNF URI, object and payload digests, variable map,
projection, full scope, producer runtime, and resource budget. Raw proof
storage and assignment payloads remain unverified when created. See
[SAT artifact contracts](../reference/capabilities/sat-smt/sat-artifacts.md).

The assignment-checker checkpoint was implemented on 2026-07-26.
`sat.model.verify` is installed only with an operator-authorized bundled
checker. Its standard-library-only clean-process implementation independently
validates the canonical CNF and all source, projection, assignment, evidence,
and lineage bindings before evaluating every clause. Acceptance uses the
runtime's existing `VerificationRecord` path. Rejection, malformed input, and
operational failure remain `UNKNOWN` and cannot establish UNSAT.

The CaDiCaL producer checkpoint was implemented on 2026-07-26.
`sat.model.find` and `sat.unsat_proof.find` are installed only when the exact
pinned CaDiCaL 3.0.1 executable is available. The adapters recheck its
executable digest, execute the deterministic DIMACS projection under enforced
wall-time and optional conflict limits, validate the competition status
protocol, and materialize only a total model or bounded raw text DRAT artifact.
Solver status, failure to produce the requested evidence, timeout, malformed
output, and stored proof bytes remain unverified and carry `UNKNOWN`. See
[SAT artifact contracts](../reference/capabilities/sat-smt/sat-artifacts.md#cadical-exploration).

The DRAT-trim checker checkpoint was implemented on 2026-07-26.
`sat.unsat_proof.verify` is installed only with bundled references and an
operator-provenanced DRAT-trim `v05.22.2023` executable. The runtime digest and
exact upstream source commit are bound into checker authorization, rechecked
at selection and around clean-process replay, and included in the verification
environment. The independent checker reconstructs canonical DIMACS and
validates certificate, payload, binding, lineage, and admitted text-proof
syntax before bounded DRAT replay. Only exit zero with exactly one
`s VERIFIED` creates a `VerificationRecord`; mutation, concatenation,
cross-CNF replay, warnings, excessive output, timeout, and runtime replacement
fail closed as `UNKNOWN`. See
[UNSAT proof verification](../reference/capabilities/sat-smt/sat-artifacts.md#unsat-proof-verification).

CaDiCaL has a small source build and a command line that accepts DIMACS plus a
proof path. DRAT-trim independently validates a DRAT proof against the input
formula. Later, evaluate whether a compatible emission or conversion path to a
maintained verified checker actually exists. Do not assume format
compatibility or block the first independent proof-checker slice on that
hardening.

Write attack tests before authorizing the UNSAT checker:

- flip a satisfying literal;
- omit, add, reorder, or renumber a bound clause;
- truncate, concatenate, or mutate a proof;
- replay a valid proof against a different CNF or variable map;
- force timeout, cancellation, excessive output, nonzero exit, and checker
  crash; and
- verify that every case fails closed without a mathematical conclusion.

Use `BOOL-MUS-001`, a small satisfiable instance, and a pigeonhole UNSAT
instance as public reproductions. Core extraction and shrinking are later
operations; first prove that the model/proof artifact boundary is sound.

The public-reproduction and held-out-evaluation checkpoint was implemented on
2026-07-26. The real CaDiCaL-to-checker reproductions pass for all three public
cases and remain explicitly unscored. After interface tuning was frozen, a
private four-pair pilot passed 4/4 under both direct control reasoning and the
SAT portfolio. All treatment runs preserved exact producer-to-verifier traces
and independently replayed, with zero false certification or tool error.
Treatment cost was materially higher, and the small sample demonstrates
assurance value rather than completion or efficiency lift. Retain the four
atomic outcomes while prioritizing compact catalog discovery over another SAT
operation. See
[the Jacobian-enabled workflow observation boundary](../reference/evaluations/evaluation-methods.md#workflow-observations).

## Wave 3: theory-bounded SMT proof slice

Begin with:

- `smt.unsat_proof.find`, backed by cvc5 and a pinned SMT-LIB profile; and
- `smt.unsat_proof.verify`, backed by a separately installed compatible
  Carcara checker.

The cvc5 Alethe documentation currently lists equality with uninterpreted
functions, linear arithmetic, bit-vectors, and parts of strings, with or
without quantifiers. The descriptor must advertise the narrower intersection
actually exercised by the pinned cvc5 and checker versions. Start with
quantifier-free EUF and linear rational or integer arithmetic. A proof
containing an unsupported rule, hole, theory, or checker-version mismatch is
unverified.

Model production and checking should be separate later slices. SMT model
semantics become subtle for arrays, functions, quantifiers, algebraic values,
and partial interpretations, so do not bundle a broad `smt.solve` result into
the first slice.

Keep Z3 as an existing exploration and differential provider. Its broad tactic
surface has uneven proof support, so a Z3 result must not verify itself.

The cvc5 producer checkpoint was implemented on 2026-07-26. The optional
`cvc5==1.3.4` wheel now exposes `smt.unsat_proof.find` for exact single-query
`QF_UF`, `QF_LIA`, and `QF_LRA` inputs. It runs in a bounded isolated worker,
materializes the exact SMT-LIB source and bound raw Alethe bytes, reports
lexical holes, and always returns `conclusion: UNKNOWN`. Public reproductions
found a zero-hole equality contradiction in `QF_UF` and explicit holes in
small linear integer and rational contradictions. This is useful producer
evidence and a concrete compatibility target for item 10, not a broad
proof-support claim. See
[SMT Alethe artifact contracts](../reference/capabilities/sat-smt/smt-artifacts.md).

## Wave 4: shared exact mathematics

### Python-FLINT first

The rational solution, rational inconsistency-certificate, and integer row-HNF
slices are implemented; see the
[exact rational linear-system evidence contract](../reference/capabilities/linear-algebra/linear-rational-solutions.md)
and [integer matrix HNF contract](../reference/capabilities/matrix/matrix-hermite-normal-form.md).
The rational-solution usability evidence is recorded in the
[the committed Harbor task boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
The remaining sequence stays demand-gated.

Implement one vertical slice at a time:

1. `linear.rational_solution.find` returns one exact vector for a declared
   rational system. `linear.rational_solution.verify` independently checks
   `A x = b`. Failure to find a vector says nothing about consistency.
2. `linear.rational_inconsistency.find` is implemented with a normalized
   left-nullspace witness `y^T A = 0`, `y^T b = 1`;
   `linear.rational_inconsistency.verify` independently replays the exact
   certificate. Failure to produce or accept a witness remains `UNKNOWN`.
3. `matrix.normal_form.hermite` is implemented with the binding's complete
   left transformation. Independent replay checks `H = U A`, unimodularity,
   and every FLINT row-HNF condition.
4. `matrix.determinant.verify` is implemented as a small shared exact
   primitive. It checks the existing SymPy determinant artifact by
   standard-library rational Gaussian elimination in an independent clean
   process; see the
   [exact rational determinant contract](../reference/capabilities/matrix/matrix-rational-determinant.md).
5. `polynomial.factor.compute` is implemented with a product relation.
   Checking that the factors multiply to the input does not by itself certify
   irreducibility or completeness; those remain separate obligations.
6. Rigorous Arb enclosure operations are implemented as computed evidence.
   The provider's rigorous error tracking is valuable, but it is not Jacobian
   `VERIFIED` until an authorized independent implementation checks the exact
   enclosure claim.

Python-FLINT wheels cover integers, rationals, modular arithmetic, polynomials,
matrices, and real and complex ball arithmetic. This makes it a better first
shared exact provider than importing SageMath.

### SymPy second

The first typed normalization slice is implemented; see the
[typed polynomial expression normalization contract](../reference/capabilities/polynomial/polynomial-expression-normalization.md)
and the
[the committed Harbor task boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
It defines a small versioned polynomial AST with explicit symbols and
coefficients, then constructs SymPy objects internally. It never passes
unsanitized user strings to `sympify`, `parse_expr`, `lambdify`, or Python
evaluation; the [SymPy documentation][sympy-parsing] explicitly warns that
these paths use `eval`.

Any follow-up differentiation or simplification remains demand-gated and must
record assumptions, domains, branch conditions, and remaining obligations.
Symbolic integration should not be an early slice: condition sets, branch
cuts, and unevaluated results make its contract and verification boundary
substantially larger.

Avoid duplicate public IDs backed separately by FLINT and SymPy. Choose the
provider that best fits the declared domain, record it in provenance, and use
the other for differential tests where useful.

## Wave 5: formal and domain branches

Promote only a branch with a completed workflow ledger row and a public
reproduction:

| Branch | First plausible slice | Required truth boundary |
| --- | --- | --- |
| First-order logic | TPTP model finding or one derivation format through E or Vampire | Independent step checker or reconstruction for the exact supported TSTP subset |
| Second proof kernel | `metamath.proof.check` against one pinned database | Separate small verifier and exact database digest |
| Graphs | `graph.canonical_label` followed by an isomorphism witness | Direct permutation and adjacency replay; no completeness claim from generator status |
| Commutative algebra | `polynomial.groebner_basis.compute` through Singular | Check generator membership and all required S-polynomial reductions over the exact coefficient domain |
| Number theory | One PARI operation not already served by FLINT | Operation-specific certificate or explicitly unverified exact computation |
| Polyhedral geometry | One Normaliz cone or Hilbert-basis operation | Separate membership, generation, minimality, and completeness claims |
| Group theory | One GAP finite-group invariant | Domain-specific replay; GAP output alone remains computed evidence |
| Lattices | `lattice.lll.reduce` | Check exact transformation, lattice preservation, and declared reducedness conditions |
| Optimization | Rational LP candidate followed by exact certificate verification | Exact primal/dual/Farkas replay; MIP requires a compatible integer-proof format |

Choose Rocq, Agda, or Isabelle only after a pinned corpus or repeated user
workflow makes its own kernel and library ecosystem material. The first
capability should be proof checking, not arbitrary interaction.

## First implementation queue

Create one focused issue per accepted item; GitHub is the source of truth
rather than an umbrella backlog.

1. Optional-provider availability, provenance, install-tier, and measurement
   contract.
2. Lean declaration search and inspection contract spike plus public
   reproduction.
3. Paired Lean discovery evaluation; decide expose, revise, consolidate into
   examples, or stop.
4. Canonical CNF, assignment, and proof artifact schemas. Implemented; see
   [SAT artifact contracts](../reference/capabilities/sat-smt/sat-artifacts.md).
5. Pure independent SAT-assignment checker with attack tests. Implemented; see
   [SAT artifact contracts](../reference/capabilities/sat-smt/sat-artifacts.md#assignment-verification).
6. CaDiCaL model and proof-producing exploration adapters. Implemented; see
   [SAT artifact contracts](../reference/capabilities/sat-smt/sat-artifacts.md#cadical-exploration).
7. DRAT-trim clean-process checker, authorization fixture, and attack tests.
   Implemented; see
   [UNSAT proof verification](../reference/capabilities/sat-smt/sat-artifacts.md#unsat-proof-verification).
8. SAT public reproductions and held-out portfolio ablation. Implemented; see
   [the Jacobian-enabled workflow observation boundary](../reference/evaluations/evaluation-methods.md#workflow-observations).
9. cvc5 Alethe proof-production spike for quantifier-free EUF and linear
   arithmetic. Implemented; see
   [SMT Alethe artifact contracts](../reference/capabilities/sat-smt/smt-artifacts.md).
10. Carcara compatibility matrix, checker adapter, mutations, and paired
    evaluation. Implemented; see
    [SMT Alethe artifact contracts](../reference/capabilities/sat-smt/smt-artifacts.md#strict-carcara-verification)
    and the
    [the task and verifier validation boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
11. Python-FLINT rational-solution find and verify slice. Implemented; see the
    [exact rational solution contract](../reference/capabilities/linear-algebra/linear-rational-solutions.md)
    and the
    [the committed Harbor task boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
12. Python-FLINT integer-matrix row-HNF slice. Implemented; see the
    [integer matrix HNF contract](../reference/capabilities/matrix/matrix-hermite-normal-form.md)
    and the
    [the committed Harbor task boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
13. Typed expression AST and one SymPy polynomial-normalization slice.
    Implemented; see the
    [typed polynomial expression normalization contract](../reference/capabilities/polynomial/polynomial-expression-normalization.md)
    and the
    [the committed Harbor task boundary](../reference/evaluations/benchmark-contracts.md#task-and-verifier-validation).
14. Python-FLINT rational inconsistency-certificate find and verify slice.
    Implemented; see the
    [exact rational linear-system evidence contract](../reference/capabilities/linear-algebra/linear-rational-solutions.md#inconsistency-certificate).
15. Independent exact rational matrix-determinant verification. Implemented;
    see the
    [exact rational determinant contract](../reference/capabilities/matrix/matrix-rational-determinant.md).
16. Re-rank E/Vampire, Metamath, nauty, Singular, PARI, Normaliz, GAP, fplll,
    and HiGHS from accumulated workflow evidence.

Stop after each paired evaluation long enough to decide whether the next
problem is missing mathematics, poor discovery, a bad schema, opaque artifacts,
parameter friction, packaging, or agent reasoning. A backend's large API is
not a reason to expose its next function.

## Definition of done for one slice

A capability slice is ready for review when it has:

1. one coherent mathematical outcome and domain-owned typed schemas;
2. canonical input and output artifacts with exact relationships and an inline
   useful summary;
3. provider version, digest, license record, feature profile, and install tier;
4. deterministic behavior where promised and full budget accounting;
5. bounded execution, cancellation, output limits, and fail-closed status
   mapping;
6. explicit exactness, scope, completeness, evidence type, assurance, and open
   obligations;
7. an independent clean-process checker and checker-binding attack tests for
   every `VERIFIED` path;
8. public positive, negative, malformed, misbound, timeout, and unsupported
   reproduction cases;
9. measured cold installation, installed size, cold start, peak memory, and
   runtime on the supported platform;
10. descriptor examples and errors that let an agent recover without reading
    backend documentation; and
11. a held-out evaluation hypothesis, control condition, independent oracle,
    contamination record, and decision owner.

The minimum go criteria are zero false `VERIFIED` results, rejection of every
known mutated certificate, deterministic artifact bindings, and no
timeout/error-to-conclusion conversion. Beyond safety, retain or prioritize a
capability only when repeated evaluations show useful correctness,
provenance, coverage, or efficiency relative to its catalog, token,
installation, and maintenance cost.

## Updating this record

Update the research snapshot when a provider version, certificate format,
license, packaging route, or measured workflow changes the ordering. For every
change:

1. link the primary maintainer source and exact version;
2. update the workflow ledger before changing rank;
3. distinguish public reproduction from held-out evaluation;
4. replace size estimates with measurements once an implementation spike
   exists;
5. record unsupported theories and proof obligations rather than silently
   narrowing claims; and
6. move accepted implementation work into a focused GitHub issue.

## Primary backend sources

Formal systems and proof-producing solvers:

- [Lean installation][lean-install], [Elan toolchains][lean-elan],
  [Lean REPL][lean-repl], and [Pantograph][pantograph]
- [CaDiCaL][cadical] and [DRAT-trim][drat-trim]
- [cvc5 proof production][cvc5-proofs], [cvc5 Alethe output][cvc5-alethe],
  and [Carcara][carcara]
- [Z3 guide][z3-guide] and [Z3 tactic proof-support summary][z3-tactics]
- [E prover][e-prover], [Vampire][vampire], and
  [TPTP derivation format][tptp-derivations]
- [Metamath book and verifier documentation][metamath]
- [Rocq platform installation][rocq-platform],
  [Isabelle installation][isabelle-install], and
  [Agda installation][agda-install]

Exact mathematics and domain systems:

- [Python-FLINT][python-flint] and [FLINT feature overview][flint-overview]
- [SymPy installation][sympy-install] and [SymPy parsing warning][sympy-parsing]
- [SageMath installation][sage-install]
- [PARI/GP][pari], [GAP reference manual][gap],
  [Singular manual][singular], and [Macaulay2 downloads][macaulay2]
- [Normaliz][normaliz] and [PyNormaliz][pynormaliz]
- [nauty and Traces][nauty]
- [HiGHS][highs]
- [fplll][fplll]

[agda-install]: https://agda.readthedocs.io/en/latest/getting-started/installation.html
[cadical]: https://github.com/arminbiere/cadical
[carcara]: https://github.com/ufmg-smite/carcara
[cvc5-alethe]: https://cvc5.github.io/docs/latest/proofs/output_alethe.html
[cvc5-proofs]: https://cvc5.github.io/docs/latest/proofs/proofs.html
[drat-trim]: https://github.com/marijnheule/drat-trim
[e-prover]: https://www.eprover.org/
[flint-overview]: https://flintlib.org/doc/overview.html
[fplll]: https://github.com/fplll/fplll
[gap]: https://docs.gap-system.org/doc/ref/manual.pdf
[highs]: https://highs.dev/
[isabelle-install]: https://isabelle.in.tum.de/installation.html
[lean-elan]: https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Managing-Toolchains-with-Elan/
[lean-install]: https://lean-lang.org/install/manual/
[lean-repl]: https://github.com/leanprover-community/repl
[macaulay2]: https://macaulay2.com/Downloads/
[metamath]: https://us.metamath.org/downloads/metamath.pdf
[nauty]: https://pallini.di.uniroma1.it/
[normaliz]: https://github.com/Normaliz/Normaliz
[pantograph]: https://github.com/leanprover/Pantograph
[pari]: https://pari.math.u-bordeaux.fr/
[pypi-json]: https://docs.pypi.org/api/json/
[pynormaliz]: https://github.com/Normaliz/PyNormaliz
[python-flint]: https://github.com/flintlib/python-flint
[rocq-platform]: https://rocq-prover.org/platform
[sage-install]: https://doc.sagemath.org/html/en/installation/index.html
[singular]: https://www.singular.uni-kl.de/index.php/singular.pdf
[sympy-install]: https://docs.sympy.org/latest/install.html
[sympy-parsing]: https://docs.sympy.org/latest/modules/parsing.html
[tptp-derivations]: https://tptp.org/UserDocs/QuickGuide/Derivations.html
[vampire]: https://vprover.github.io/
[z3-guide]: https://microsoft.github.io/z3guide/
[z3-tactics]: https://microsoft.github.io/z3guide/docs/strategies/summary/
