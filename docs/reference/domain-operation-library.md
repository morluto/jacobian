# Domain operation library

[Documentation home](../index.md)

- Status: Current implementation reference; capability contracts are pre-stable
- Related architecture:
  [Domain operation library](../explanation/architecture.md#domain-operation-library)

Built-in mathematical producers are grouped into explicit, domain-owned
bundles. The library removes repeated adapter plumbing while preserving the
public capability contract: each operation still has one typed mathematical
outcome, material artifacts and relationships, declared scope and
completeness, provider provenance, and visible proof obligations.

This document describes the shared installation contract. It is not a static
capability inventory. Read `capability://catalog`, then use
`capability.describe` for the IDs, schemas, modes, provider identities, and
checker requirements installed in a particular runtime.

## Bundle contract

A `DomainBundle` declares:

- a domain and schema namespace;
- content-addressed mathematical semantics;
- a measured provider runtime and backend version;
- domain wording for invalid requests, scope, completeness, and assurance; and
- an explicit tuple of `ComputedOperation` or `BoundedSearchOperation`
  declarations.

Bundles are imported through the fixed built-in portfolio. They do not use a
global registry, recursive package scan, compatibility wrapper, or import-time
registration. Runtime registration still applies configured exclusions and
provider health checks uniformly.

The current built-in domains cover arithmetic, number theory, combinatorics,
finite sets, sequences, Euclidean and projective geometry, graph optimization
and invariants, matrices, lattices, polynomials, validated real analysis,
finite probability, and rational optimization. Catalog membership remains the
authority for actual availability. The current portfolio also includes finite
simplicial-complex materialization, oriented chain complexes, and prime-field
homology under the `topology` domain, plus canonical finite partial orders,
Dilworth width certificates, complete ideal recurrences, and Möbius values
under the `poset` domain.

## Computed operations

A `ComputedOperation` couples Pydantic request and result models with a
deterministic finite implementation. Its implementation returns exactly one of:

| Outcome | Meaning |
| --- | --- |
| `ComputedSuccess` | A complete, contract-valid result candidate |
| `ComputedNotApplicable` | A valid request outside the mathematical domain |
| `OperationExecutionFailure` | `ERROR`, `TIMEOUT`, or `CANCELLED`, with no mathematical conclusion |

The installer validates the complete request model before computation or
artifact writes. On success it validates the returned result again,
materializes input and result artifacts, records their lineage and relation,
and returns `COMPLETE · COMPUTED`. Neither exact arithmetic nor deterministic
execution grants `VERIFIED`.

## Bounded searches

A `BoundedSearchOperation` adds a completion predicate, a typed scope
projection, an optimality-obligation model, and an explicit basis for unknown
completeness. Its outcomes distinguish:

| Outcome | Execution and mathematical meaning |
| --- | --- |
| `BoundedSearchWitness` | The declared completion predicate holds; the result is still only `COMPUTED` |
| `BoundedSearchIncomplete` | Execution completed with a useful partial result; completeness is `UNKNOWN` |
| `BoundedSearchInterrupted` | A timeout, cancellation, or error preserved a partial result at heuristic assurance |
| `BoundedSearchNotApplicable` | The valid request lies outside the operation's domain |

Every materialized search result carries its input, result, declared scope, and
an open optimality-obligation artifact. Incumbents, lower or upper bounds, and
traces remain inspectable when the search is incomplete. An execution status of
`COMPLETED` says that the implementation returned normally; it does not close
the optimality obligation or establish a conclusion.

The adapter rejects an implementation whose outcome variant contradicts its
completion predicate. Timeouts, cancellations, errors, resource exhaustion,
and failure to find a witness remain non-conclusions.

## Artifacts and assurance

`OperationInstaller` owns the generic mechanics:

1. register the bundle semantics and request, result, and obligation schemas;
2. validate the full request before invoking domain code;
3. validate the returned result and obligation;
4. materialize content-addressed artifacts with explicit parents;
5. project relationships, scope, completeness, diagnostics, and provenance
   into `CapabilityResult`; and
6. cap producer assurance at `COMPUTED`, or `HEURISTIC` after interruption.

Domain implementations therefore depend on mathematical contracts and
maintained libraries, not artifact stores, capability envelopes, or checker
authorization.

## Independent exact replay

Some polynomial, matrix, graph, geometry, probability, topology, poset, and
combinatorics results have a separate verification capability. The producer
first returns a result artifact in `EXPLORE` mode. A verification request then
supplies that exact `result_uri` to the matching `VERIFY` capability.

Domain-owned `ExactReplayCheckerDeclaration` values name the request model,
certificate format, and checker function, but they carry no authority.
Operator-controlled installation creates checker registry entries with exact
claim-schema, semantics, candidate-schema, evidence-format, and provider
identity allowlists. Verification re-resolves the input and result lineage,
remeasures the checker runtime, and replays the mathematical relation in a
bounded worker.

Built-in producer bundles are installed independently of checker
authorization. In the reference runtime, exact replay capabilities are present
only when bundled references are enabled; disabling them leaves the computed
producers available but removes those verification capabilities from the
catalog. Operator-installed packages follow the same separation between
availability and authorization.

A successful replay returns a verification record bound to the operation,
input artifact, result artifact, witness, checker identity, semantics, and
format. Unsupported formats, rejected evidence, timeout, cancellation,
runtime-identity drift, malformed output, and worker failure return
non-verifying outcomes and cannot create that record.

Use `capability.describe` on the producer and candidate verification capability
before invocation. Do not infer a verifier ID or payload from naming
conventions: checker availability depends on operator authorization and the
installed runtime.

Two bounded portfolio examples show the intended boundary:

- `graph.hamiltonian_path.decide` returns either a complete spanning path
  witness or a negative decision after exhausting its order-18 state space.
  `graph.hamiltonian_path.verify` checks a positive witness directly and
  independently exhausts negative instances.
- `polynomial.jacobian_syzygy.minimum_degree.compute` constructs the graded
  maps from degree zero through the first kernel or a declared finite bound.
  The source can be a canonical sparse polynomial or a labelled product of
  rational linear forms; the latter keeps factor-to-expansion provenance inside
  the producer and checker boundary.
  Its compact default exposes bases, map digests, ranks, nonzero minors, and a
  kernel witness; full sparse entries are opt-in.
  `polynomial.jacobian_syzygy.minimum_degree.verify` reconstructs the maps with
  a standard-library rational checker independent of the SymPy producer.

`geometry.projective_line_arrangement.flats.materialize` is a complete finite
materializer rather than a theorem prover. It normalizes labelled rational
projective lines, groups every pair intersection, recovers all incidences, and
reports multiplicity and pair accounting at `COMPUTED` assurance. When the
operator authorizes it,
`geometry.projective_line_arrangement.flats.verify` independently replays all
of those finite incidence obligations and returns the bound verification
record.

## Adding an operation

Keep additions domain-owned and follow the nearby bundle:

1. define bounded Pydantic request, result, and, for search, obligation models;
2. implement one mathematical outcome without store or envelope dependencies;
3. declare a computed or bounded-search operation in a subject module;
4. include it explicitly in the domain bundle and declare the maintained
   provider runtime and backend version;
5. test request validation, artifacts and lineage, assurance, failure
   semantics, and catalog projection; and
6. add an independent checker only when the exact relation has a separate,
   operator-authorized replay path.

Do not add a mechanical wrapper for every backend function, hide a research
workflow in one operation, authorize a checker from domain code, or interpret
missing witnesses and incomplete searches as negative conclusions.
