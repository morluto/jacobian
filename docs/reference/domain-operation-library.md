# Domain operation library

[Documentation home](../index.md)

- Status: Current implementation reference; capability contracts are pre-stable
- Related architecture:
  [Domain operation library](../explanation/architecture.md#domain-operation-library)

Built-in mathematical producers are grouped into explicit, domain-owned
bundles. The library removes repeated adapter plumbing while preserving the
public capability contract: each operation still has one typed mathematical
outcome, optional durable artifacts and relationships, declared scope and
completeness, provider provenance, and visible proof obligations.

This document describes the shared installation contract. It is not a static
capability inventory. Read `capability://catalog`, then use
`math.find` for the IDs, schemas, modes, provider identities, and
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
simplicial-complex materialization, oriented chain complexes, prime-field
homology, and transformation-certified integral homology under the `topology`
domain. A dedicated `certified_snf` bundle supplies bounded full left/right
Smith basis changes without replacing the existing diagonal-only matrix
outcome. The portfolio also includes canonical finite partial orders, Dilworth
width certificates, complete ideal recurrences, and Möbius values under the
`poset` domain.

## Computed operations

A `ComputedOperation` couples Pydantic request and result models with a
deterministic finite implementation. Its implementation returns exactly one of:

| Outcome | Meaning |
| --- | --- |
| `ComputedSuccess` | A complete, contract-valid result candidate |
| `ComputedNotApplicable` | A valid request outside the mathematical domain |
| `OperationExecutionFailure` | `ERROR`, `TIMEOUT`, or `CANCELLED`, with no mathematical conclusion |

The installer validates the complete request model before computation. On
success it validates the returned result again and returns it inline as
`COMPLETE · COMPUTED`, without generic input or result artifacts and without an
episode. Neither exact arithmetic nor deterministic execution grants
`VERIFIED`. A reusable mathematical object or evidence-bearing result belongs
in an explicitly artifact-producing capability, not behind a persistence flag
on an ordinary computation.

### Static type contract

The generic operation layer preserves each domain contract through
construction and installation:

```python
ComputedOperation[RequestT, ResultT]
ComputedOperationFactory(
    operation: Callable[[RequestT], ResultT],
) -> ComputedOperation[RequestT, ResultT]
BoundedSearchOperation[RequestT, ResultT, ObligationT]
```

Domain implementations accept their concrete validated request model and
return their concrete result model. Bounded searches likewise return their
declared obligation model from the obligation builder. A broad
`Callable[[ContractModel], ContractModel]` boundary, or a cast from
`ContractModel` to the declared request type inside an implementation, defeats
this contract and is not supported.

This static precision does not replace runtime validation. The installer still
validates untrusted input with the declared Pydantic request model before it
calls domain code, then validates returned result and obligation values before
returning them or performing any explicit artifact writes.

## Native Python relationship

The supported native-value interface is the deliberately small
[`jacobian.math`](python-api.md) namespace. Its functions accept Python,
SymPy, or NetworkX values as documented and call typed mathematical kernels
directly. They do not construct the capability runtime or route through
`math.run`.

When a native function corresponds to a capability, both paths share the same
domain-owned mathematical kernel. Explicit adapters translate between native
values and the existing Pydantic request and result contracts. This keeps one
mathematical implementation while preserving capability schemas, completeness,
provenance, verification behavior, and artifact lineage when artifacts are
part of the outcome. Generic
reflection, automatic model generation, and universal value conversion are
outside this library's contract.

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

## Values, artifacts, and assurance

`OperationInstaller` owns the generic mechanics:

1. register the bundle semantics and request, result, and obligation schemas;
2. validate the full request before invoking domain code;
3. validate the returned result and obligation;
4. return ordinary computed results inline, or delegate explicit durable
   outcomes to their artifact-producing capability;
5. project relationships, scope, completeness, diagnostics, and provenance
   into `CapabilityResult`; and
6. cap producer assurance at `COMPUTED`, or `HEURISTIC` after interruption.

Domain implementations therefore depend on mathematical contracts and
maintained libraries, not artifact stores, capability envelopes, persistence
flags, or checker authorization.

## Independent exact replay

Exact replay also covers bounded sparse rational-function identities. The
identity verifier preserves the submitted numerators and denominators and
checks fraction-field equality by independent polynomial cross multiplication;
pointwise denominator-definedness remains outside its scope. See
[Rational-function identities](capabilities/polynomial/rational-function-identities.md).

Some polynomial, matrix, graph, geometry, probability, topology, poset, and
combinatorics results have a **separate checker tool** (distinct catalog ID).
The producer returns a result (and often a `result_uri`); the agent then runs
the matching `*.verify` (or equivalent) tool with that exact lineage. Do not
treat this as switching `EXPLORE`/`VERIFY` mode on one ID—see
[#1143](https://github.com/morluto/jacobian/issues/1143).

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

Use `math.find` on the producer and candidate verification capability
before invocation. Do not infer a verifier ID or payload from naming
conventions: checker availability depends on operator authorization and the
installed runtime.

Bounded portfolio examples show the intended boundary:

- `graph.hamiltonian_path.decide` returns either a complete spanning path
  witness or a negative decision after exhausting its order-18 state space.
  `graph.hamiltonian_path.verify` checks a positive witness directly and
  independently exhausts negative instances.
- `combinatorics.integer_set.sidon.decide` materializes the complete ordered
  integer-difference profile. `combinatorics.cyclic_difference_set.perfect.decide`
  similarly materializes every nonzero cyclic residue multiplicity.
  `combinatorics.cyclic_difference_set.extension.decide` asks only a bounded,
  fixed-order direct-containment question in the derived modulus `k(k-1)+1`;
  its negative result is durable and records the exact candidate-space size.
  The corresponding `.verify` capabilities use a standard-library checker
  module that imports no producer code. Negative extension replay enumerates
  every candidate with `itertools.combinations`, independently of the
  producer's pruned depth-first search.
- `polynomial.jacobian_syzygy.minimum_degree.compute` constructs the graded
  maps from degree zero through the first kernel or a declared finite bound.
  The source can be a canonical sparse polynomial or a labelled product of
  rational linear forms; the latter keeps factor-to-expansion provenance inside
  the producer and checker boundary.
  Its compact default exposes bases, map digests, ranks, nonzero minors, and a
  kernel witness. This compact producer accepts the `CERTIFICATES` request
  detail; requests with `coefficient_map_detail: "SPARSE_ENTRIES"` belong to
  the explicit `polynomial.jacobian_syzygy.coefficients.materialize` capability,
  which retains the complete sparse coefficient ledger.
  `polynomial.jacobian_syzygy.minimum_degree.verify` reconstructs the maps with
  a standard-library rational checker independent of the SymPy producer.
- `polynomial.jacobian_degree_slice.system.materialize` writes the frozen
  normalized bivariate degree-`(2,3)` system as a producer-only typed artifact.
  Exact degree is the disjunction that the quadratic and cubic top coefficient
  vectors are nonzero; the materializer represents it by the complete twelve
  charts `t*a_i*b_j-1`, not by requiring every leading coefficient to be
  nonzero. `polynomial.nullstellensatz.infeasibility_certificate.compute` is
  installed only with pinned Singular 4.4.1p5 and persists a bounded bundle of
  identities `sum(h_i*f_i)=1` under the exact system semantics. The bundle is a
  child of the system artifact and records its URI and object digest.
  `polynomial.nullstellensatz.infeasibility_certificate.verify` independently
  reconstructs the frozen generators and replays sparse rational products in
  a standard-library-only checker. The producer and checker share neither
  Gröbner reduction nor certificate-generation code. Missing Singular removes
  only the producer; missing checker authority leaves materialization usable
  and makes verification return a non-conclusion.

The degree-slice system and certificate bundle use producer-only schemas rather
than a new global persistence enum. This gives typed artifact handoff and
durable lineage now, while leaving the broader persistence-policy design in
issue #386 unresolved. Consumers must pass the returned artifact URIs; inline
summaries are not substitutes for the stored system or certificate.

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
2. implement one mathematical outcome with concrete request, result, and
   obligation types and without store or envelope dependencies;
3. declare a computed or bounded-search operation in a subject module;
4. include it explicitly in the domain bundle and declare the maintained
   provider runtime and backend version;
5. test request validation, artifacts and lineage, assurance, failure
   semantics, and catalog projection; and
6. if the outcome belongs in the supported native Python API, share the typed
   kernel, add explicit domain-owned conversions, and update the public symbol
   manifest and import-isolation tests; and
7. add an independent checker only when the exact relation has a separate,
   operator-authorized replay path.

Do not add a mechanical wrapper for every backend function, hide a research
workflow in one operation, authorize a checker from domain code, or interpret
missing witnesses and incomplete searches as negative conclusions.

A new built-in domain changes its own package and the explicit ordered factory
tuple. It does not change generic MCP tools, checker authority, shared
documentation landing-page registries, `tests/topology.toml`, or
`.github/ci-impact.json`. The latter two remain CI-owned control planes.
