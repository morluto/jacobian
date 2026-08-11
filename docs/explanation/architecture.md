# Architecture

[Documentation home](../index.md)

- Status: Current architecture

## Purpose

The [product model](product-blueprint.md) defines Jacobian as a toolbox of
atomic math tools behind a fixed search/execute surface. This document
describes the runtime, ownership boundaries, and trust zones that support that
model.

```text
agent
  ├── math.find ──► discover / inspect math tools
  └── math.run  ──► one named tool → mathematical value or checker verdict
```

Ordinary tools return **calculations** (values) plus execution status. Checker
tools are **additional catalog IDs** that return a check **verdict**; they are
never a second role on the same tool. Independent checkers are
operator-authorized packages, not self-certified producer success.

Models, search algorithms, and domain solvers may be heuristic or incomplete.
That does not change the product rule: return the math; use a separate tool
when independent checking is required.

When a checker tool accepts a claim, it may persist a bound verification
record. Schema validation alone does not establish informal↔formal
correspondence or a theorem.

## Ownership boundaries

The runtime owns catalog install, execution bounds, artifact identity where
used, and operator authorization of checker packages—not mathematical strategy.
Adapters expose external systems as catalog tools. Domains own schemas and
kernels. Checker packages implement independent replay and never self-authorize.

Agents own multi-step strategy. A tool may coordinate backend calls only for
**one** agent-visible outcome; useful intermediate values stay visible. Worked
examples belong in scenarios and benchmarks, not generic runtime types.

## Search and execute

Two agent verbs over a growing catalog of **atomic math tools** (API name:
*capabilities*). Progressive disclosure of operations—not one MCP tool per
domain, not a code-execution sandbox, not explore/verify modes.

```text
Agent sees (fixed surface)
  math.find                 search / inspect math tools
  math.run                  execute one math tool
  capability://catalog      full installed inventory when needed

Host owns (three zones)
  catalog / search          what exists and how to find it
  execute                   one named run → value or verdict
  host plumbing             process, store, install facts, lifecycle
```

| Verb | MCP tool | Meaning |
| --- | --- | --- |
| Search | `math.find` | Find or inspect tools by query, browse, or exact ID (schemas, examples). |
| Execute | `math.run` | Run one tool by ID. Primary payload is the **mathematical result** (or checker **verdict**). |

The catalog grows; the MCP tool count does not. Multi-step work is several
find/run turns. The runtime does not prescribe order or recommend next steps.

The [tool surface reference](../reference/tools.md) defines current schemas and
projections.

### What this is not

- A new top-level MCP tool per mathematical operation or paper.
- Dual-mode tools (one ID that both computes and independently verifies).
- Code Mode-style host orchestration via generated sandbox scripts.
- A planner with “recommended next tool” rankings.
- A product whose main return type is an assurance slogan rather than a value.

### Catalog and search zone

Owns what is installed and how agents discover it—not running mathematics.

- Descriptors: ID, version, description, tags, provider identity, schemas.
- Compact search cards and inspect-by-ID full contracts.
- Shared mathematical object vocabulary as it becomes first-class.
- Portfolio growth via reusable primitives, not paper-shaped mega-tools.
- Honest retrieval (lexical today; typed applicability later without policy).

`capability://catalog` is the full inventory. `math.find` is progressive
discovery.

### Execute zone

Owns one named invocation and its typed result—not research strategy.

```text
math.run
  → resolve tool by ID
  → validate request
  → run kernel (ordinary tool) or authorized checker (checker tool ID)
  → return mathematical value or checker verdict + execution status
  → attach only explicitly declared artifact references
```

- Ordinary tools: primary output is the **calculation**; they do not become
  formal theorems by succeeding.
- Checker tools: **separate IDs**; verdict from operator-authorized independent
  replay only.
- Timeout, cancel, error, invalid input: non-conclusions.
- No dual-mode tools; checkers are separate catalog IDs.

One `math.run` may call several backends only when they implement **one**
agent-visible outcome.

### Host plumbing zone

Owns process boundaries, durable state, provider inventory, and lifecycle—not
mathematical meaning.

- Shared worker harness; lifecycle state machines; inventory ≠ install ≠
  checker auth; lazy probes; content-addressed artifacts and lock budgets.

### Domain and checker contributions

Domains: request/result models, kernels, catalog registration for ordinary
tools. Checkers: independent replay as **additional** catalog entries.
Neither adds a private MCP tool nor a second hand-written wire language for the
same contract.

```text
tool_id, version
request / result models     source of truth
primary result              math value OR checker verdict
object types, cost bounds
provider requirements       host inventory
```

Search projects that packet. Execute enforces it. Host installs and bounds it.

### Organizing changes

| Change | Zone |
| --- | --- |
| How agents discover or inspect operations | Catalog / search |
| What a single run means, returns, or may claim | Execute |
| Process, store, install, or lifecycle without math semantics | Host plumbing |
| New top-level MCP tool or encoded research strategy | Reject |
| Second hand-written copy of an existing contract | Reject or generate from the authoritative model |

## Trust zones

### Trusted inputs and services

- Versioned claim schemas
- Versioned domain semantics
- The checker registry
- Operator-authorized witness, certificate, and transformation checkers
- The artifact identity and certificate-binding implementation

### Untrusted accelerators

- Candidate generators and mutators
- Heuristic and exact-candidate evaluators
- Witness oracles
- Structure enumerators and canonicalizers used for search
- Representation transformers
- SAT, SMT, LP, MIP, and polyhedral solvers
- Language-model output
- Operator-installed plugin code as a source of mathematical claims

A solver or evaluator can produce evidence. It cannot promote its own evidence
to `VERIFIED`.

## Core records

Jacobian separates four kinds of state.

### Mathematical object

An immutable object encoded using a versioned schema and canonicalizer.

```text
object_digest = SHA256(
    object_format_version
    || schema_uri
    || semantics_uri
    || canonicalizer_digest
    || canonical_bytes
)
```

This domain-separated digest prevents the same bytes from silently acquiring a
different meaning under another schema or semantics version.

### Artifact manifest

An immutable, content-addressed record connecting an object to its media type,
schema, parent artifacts, and a short summary.

Object identity and artifact identity answer different questions. The object
digest binds canonical payload, schema, semantics, and canonicalizer. The
artifact URI also binds carrier metadata such as parents and summary. Code that
authorizes replay or promotion must require the exact artifact URI when lineage
matters; equal object digests do not make two carriers interchangeable.

### Run record

Execution metadata such as runtime, seed, environment, limits, logs, and tool
version. A run record does not change the identity of the mathematical object.
Long-running capability adapters may additionally persist append-only lifecycle
events, immutable checkpoints, archive pages, and archive manifests around a
small mutable snapshot index.

## Model-facing capability API

The agent product shape is [search and execute](#search-and-execute):
`math.find` discovers math tools; `math.run` executes one tool. This section
describes the registry and installation machinery behind that pair.

`CapabilityService` is a registry of operator-installed adapters exposing a
broad portfolio of mathematical capabilities. Each adapter declares a
namespaced operation ID, version, input and output JSON Schemas, and discovery
metadata. (Legacy wire fields may still list a single `mode` per tool until
[#1143](https://github.com/morluto/jacobian/issues/1143) lands.) The MCP projection exposes the
installed descriptors through `capability://catalog` and the tool-callable
`math.find` and `math.run` pair, so a new Alloy, Lean,
SAT/SMT, CAS, or domain adapter does not require another MCP tool or a
generic-core type. The current catalog returns full descriptors. As the
portfolio grows, discovery should keep progressive search (`math.find`) and
compact cards strong enough that agents need not load every schema into the
initial context. Domain descriptions project exact schemas, binding rules, and
executable examples without moving mathematical semantics into the generic
runtime.

Registration also enforces the
[provider runtime contract](../reference/provider-runtime.md). The catalog
contains only adapters whose exact source tree, Python distribution manifest,
or executable identity is available and healthy. Descriptor metadata records
the provider version, digest coverage, platform, installation tier, license,
features, and fixed checker identities. Invocation results bind the selected
provider and digest without treating operational provenance as mathematical
assurance.

The service validates both schemas and prevents adapters from self-promoting:
`VERIFIED` requires a valid local verification record whose checked evidence
is returned with the capability result. Stage-aware diagnostics separate
invalid input, reference resolution, adapter execution, and checker outcomes.

A registered capability should expose one observable mathematical operation.
Broad tasks are workflows over multiple capability invocations owned by the
agent, not by the runtime. A capability may coordinate several backend calls
when they implement that one operation. Useful intermediate values and
verification obligations remain visible; artifacts are materialized only when
durable identity, independent retrieval, replay, resumability, evidence
binding, or size-separated transport requires them.

### Mathematical value and backend layers

Shared Pydantic mathematical contracts own provider-independent value identity:
the exact parent or coefficient domain, axes, basis, ordering, canonical
representation, and other facts that change mathematical interpretation.
Backend-native objects are computational representations of those values, not
composition, wire, or persistence identities.

```text
jacobian.contracts
    shared mathematical values
             │
             ▼
jacobian.domains.<domain>
    explicit contract ↔ backend conversions
    typed mathematical kernels
             │
       ┌─────┴──────────────┐
       ▼                    ▼
jacobian.math        operation declarations
native facade                 │
                              ▼
                        math.find / math.run
```

Conversions between a shared contract and SymPy, FLINT, NetworkX, or another
maintained backend are explicit and domain-owned. The native API and catalog
operations call the same typed kernels; neither calls through the other.
Provider selection, invocation provenance, persistence, MCP transport, and
checker assurance remain outside the mathematical value. A change of backend
therefore does not change value identity unless it also changes mathematical
meaning or canonicalization.

This is a dependency direction, not a universal algebra framework. Jacobian
does not add a second semantic type system, unrestricted backend wrapper,
automatic coercion graph, or generic conversion language above its maintained
libraries. Mathematical transformations such as basis change, restriction of
scalars, quotient reduction, and nontrivial reindexing remain explicit
domain-owned operations.

### Domain operation library

Built-in mathematical producers live in explicit domain packages. A package
exports a `DomainBundle` factory; subject modules export named collections such
as `POINT_CAPABILITIES` or `DIVISIBILITY_CAPABILITIES`, and `bundle.py` combines
them without import-time instances, registration, or recursive discovery. The
single built-in composition module holds the ordered tuple of factories.

`ComputedOperation` declares a deterministic typed producer.
`BoundedSearchOperation` additionally distinguishes a complete witness from an
incomplete result and carries a typed scope projection plus the basis for
unknown completeness. Both use Pydantic request and result models.
`OperationInstaller` projects either operation into the existing
`CapabilityAdapter` protocol, registers schemas and semantics, and caps producer
assurance at `COMPUTED`. Ordinary computed values remain inline. Explicit
materializers and evidence-producing capabilities create artifacts with
lineage; bounded searches retain durable state only where their resumability or
evidence contract requires it. Domain functions therefore depend on
mathematical libraries and contracts, not stores, protocol envelopes, or
checker authorization.

The runtime builds and installs a typed, fixed `PortfolioPlan` through
`PortfolioAssembler`. Bundles declare earlier domain dependencies explicitly;
a managed installer receives only those installed dependencies. Exact replay
declarations stay on their producer bundles, while the operator-owned
verification phase alone authorizes and installs them. Its
`InstallationContext` owns the exclusion-aware registration callback, so the
runtime remains a lifecycle owner rather than a registration facade. There is
no global operation registry, recursive package scan, compatibility adapter,
or registration side effect.

The built-in portfolio spans arithmetic, number theory, combinatorics, finite
sets, sequences, geometry, graph optimization and invariants, matrices,
lattices, polynomials, validated real analysis, finite probability, and
rational optimization. Each bundle declares the provider runtime and backend
version whose identity appears in descriptors and results. The installed
catalog, rather than this list, remains the authority for availability and
exact contracts.

`AtomicServiceAdapter` has a narrower role: it projects existing stateful
services that already return rich result envelopes. Domain-owned mathematical
functions use domain operations instead. Checker-backed verification remains
specialized because authorization, evidence binding, and replay are trust
boundaries rather than producer metadata.

The [domain operation library reference](../reference/domain-operation-library.md)
defines the producer outcomes, bounded-search obligations, and independent
exact-replay path.

### Lean formal intermediates

Lean exploration exposes typed proof states, tactic transitions, bounded
dependency subgraphs, premise-retrieval provenance, and exact proof edits.
These are agent-visible intermediate objects rather than an opaque proof-repair
workflow. An accepted proof edit is bound to an independent Lean verification
record; validation or generation alone cannot accept it.

Proof repair is an agent-owned composition rather than a single opaque
capability. Agents compose proof-state inspection, premise retrieval, an exact
edit proposal, checker-backed edit validation, and `lean.check` as distinct
operations. Generated edits remain unverified unless independent replay accepts
the exact proof.

## Common result model

Operational state, mathematical conclusion, and assurance are orthogonal:

```json
{
  "execution": {
    "status": "COMPLETED",
    "runtime_ms": 1240
  },
  "input": {
    "status": "ACCEPTED"
  },
  "claim_digest": "sha256:...",
  "candidate_digest": "sha256:...",
  "conclusion": "FALSE",
  "assurance": {
    "arithmetic": "EXACT_INTEGER",
    "method": "DIRECT_WITNESS",
    "coverage": "NOT_APPLICABLE",
    "verification": "VERIFIED",
    "checker_digest": "sha256:...",
    "scope_uri": "artifact://sha256/..."
  },
  "evidence_uris": ["artifact://sha256/..."],
  "trace_uri": "artifact://sha256/..."
}
```

Required enums:

```text
execution.status:
    COMPLETED | TIMEOUT | CANCELLED | ERROR

input.status:
    ACCEPTED | REJECTED

conclusion:
    TRUE | FALSE | UNKNOWN | NOT_APPLICABLE

arithmetic:
    EXACT_INTEGER
    EXACT_RATIONAL
    EXACT_ALGEBRAIC
    VERIFIED_INTERVAL
    SYMBOLIC
    FLOATING_HEURISTIC

method:
    DIRECT_WITNESS
    EXHAUSTIVE_FINITE
    CHECKED_CERTIFICATE
    BOUNDED_SEARCH
    SAMPLING
    HEURISTIC

coverage:
    EXHAUSTIVE
    BOUNDED
    RESTRICTED
    SAMPLED
    NOT_APPLICABLE

verification:
    UNVERIFIED | VERIFIED
```

Only an operator-authorized independent checker may produce
`verification = VERIFIED`.
`TIMEOUT` and `ERROR` are execution states, not mathematical conclusions.
A verified result is not limited to rational exhaustive enumeration:
kernel-checked symbolic proofs, exact algebraic certificates, and
outward-rounded interval certificates are valid assurance mechanisms when an
authorized checker replays them. Such proof certificates may use
`coverage = NOT_APPLICABLE`; direct finite enumeration must instead report
`EXHAUSTIVE`.

## Domain capability adapters

Domains expose distinct, optional capabilities rather than implementing one
mandatory problem interface. Capability IDs are descriptive and domain-owned,
for example `graph.enumerate.nonisomorphic` or
`polynomial.compute.groebner_basis`. Their input and output contracts may use
domain-specific schemas for candidates, invariants, transformations, witnesses,
or certificates. The runtime does not impose a universal operation enum or
shared mathematical object ontology.

A domain adapter may delegate its operation to a proof assistant, CAS, solver,
database, or domain library. It may also coordinate several backend calls when
they produce one coherent mathematical result. It must return the typed result,
material artifacts, provenance, semantic limits, and any remaining proof
obligation through the common capability contract.

Search plugins cannot register themselves as trusted checkers. The checker
registry is operator-managed and binds checker digests to supported claim,
semantics, and certificate versions.

Plugins define mathematical meaning; adapters present their operations through
`math.find` and `math.run`; the runtime enforces the common
artifact and assurance contract.

### Sealed plugin identity

Installation creates one immutable registry snapshot that binds the manifest,
capability entrypoints, each implementation package digest, runtime and build
identity, and platform compatibility. Discovery inspects source files without
importing the package. Capability resolution remeasures the package before
execution, so a changed file cannot continue under the installed snapshot.
Providers that execute multiple maintained components use a composite identity
that binds every source tree or Python distribution record; resolution
remeasures every component rather than trusting the top-level package alone.

The initial package format hashes regular package files, while declared and
imported modules must be Python source. Symlinks, traversal outside the
package, bytecode-only module execution, and native extension-module execution
are rejected. This protects registry identity; it does not sandbox
operator-installed code once a worker executes it.

The generic fault matrix runs against a disposable, conformance-only package in
isolated state. Production plugins are not expected to expose inputs that
deliberately crash, hang, or emit malformed responses.

## Search and checker separation

The independent checker boundary is unchanged by search, transformation, or
other capability adapters. The independent checker may share stable wire
schemas and primitive exact arithmetic types with the search side. It must not
import candidate-generation, search, canonicalization, or solver
implementations. Search, generation, evaluation, and transformation output
never self-certifies;
`VERIFIED` requires an operator-authorized checker independent of the
proposing, searching, or evaluating implementation.

Higher assurance may add a second implementation using a different algorithm or
a proof-assistant kernel. Different programming languages are useful for
defense in depth, but language diversity alone does not establish mathematical
independence.

Domain-owned exact replay declarations do not authorize themselves.
Operator-controlled installation translates each declaration into a checker
registry entry with explicit schema, semantics, evidence-format, candidate,
and provider-runtime compatibility. The producer remains capped at `COMPUTED`;
the separately installed verification capability may return `VERIFIED` only
after replay creates a fully bound verification record.

Bounded workers, including checker replay, execute outside the control process
with wall-time and output limits. Supported POSIX environments additionally
apply CPU and address-space limits before execution. Checker dispatch
remeasures the authorized provider identity, and an interrupted or
identity-mismatched replay cannot create a verification record.

## External process ownership

`run_bounded_process` is the low-level engine for bounded capture, deadlines,
cancellation, resource limits, and descendant cleanup. Product code reaches it
through the product process-policy gateway, which requires an absolute primary
executable, explicit working directory and environment, positive timeout, and
stdout and stderr limits. Provider bootstrap owns executable discovery and
records the resolved identity; the gateway never searches ambient `PATH`.
Authorized provider and checker identities are remeasured immediately before
execution.

Repository operations use a separate tooling command runner. Harbor, Git,
Docker, `uvx`, validation commands, and feasibility probes resolve their
operator-installed executables before constructing a bounded request. Product
runtime code must not import this runner. Both boundaries use explicit
environment policies rather than forwarding the complete host environment;
providers that launch nested tools receive only an authorized toolchain
`PATH`.

These boundaries provide bounded host-process containment. They do not claim
filesystem or network sandboxing, and process completion never establishes a
mathematical conclusion.

## Bounded discovery

A bounded enumeration capability validates its claim, domain contract, scope,
and implementation identity before creating a durable experiment handle. Its
adapter may page through the declared scope, commit candidate and evaluation
artifacts, and maintain exact accounting:

```text
enumerator page
    → schema validation
    → optional implementation-bound canonical key
    → duplicate rejection
    → batch evaluation
    → immutable archive page
    → durable snapshot
```

The snapshot distinguishes complete enumerator reports, candidate limits,
wall-time limits, cancellation, and errors. Even a complete report remains
unverified. Canonical mathematical objects retain ordinary artifact identity;
the search key separately hashes the canonical object digest together with the
canonicalizer implementation digest.

Short bounded mathematical searches use the same fail-closed distinction
without creating a durable experiment. They preserve the typed incumbent,
bounds, trace, declared scope, and an open optimality obligation. A search may
finish operationally with a useful partial result while completeness remains
`UNKNOWN`; `COMPLETED` therefore describes execution, not optimality or a
mathematical conclusion. Timeout, cancellation, and worker failure retain
inspectable partial artifacts at heuristic assurance.

Representation-changing capabilities follow a producer/checker split. A
transformer stores the target, relation label, implementation digest, and proof
obligation. An independently checked invocation rebinds both source and target
schemas, semantics, and digests before dispatching to an operator-authorized
checker.

For example, a polytope separation adapter may ask Z3 to propose exact convex
weights or an exact separator. An independent checker can replay that object
using exact rational arithmetic without importing Z3.

## Durable capability execution

Long-running tools (for example bounded search) may use a durable experiment
runtime. This is still ordinary `math.find` / `math.run`—not a separate MCP
workflow or agent research mode:

```text
idempotent request
    → SQLite acceptance row + append-only event
    → bounded adapter work in child processes
    → immutable archive page + checkpoint
    → atomic snapshot update
    → pause, resume, or terminal archive
```

An idempotency key binds one exact request digest to one experiment URI.
Concurrent submissions of that request reuse the accepted experiment; the same
key cannot be rebound to another request. Plugin work performed after the last
checkpoint may run again after process loss, but only committed pages and
checkpoints become durable lineage.

On startup, active experiments are changed to `PAUSED`, while pending
cancellation becomes `CANCELLED`. A malformed snapshot is moved to `ERROR` and
recorded in `search_recovery_failures` without preventing unrelated rows from
recovering. Checkpoint restoration rebinds the request, plugin snapshot,
implementation digests, effective budget, environment, archive pages, and
accounting before opaque adapter state is accepted.

The local scheduler accepts one worker and requires one active Jacobian process
per state directory. SQLite provides transactional request acceptance, not a
distributed worker lease.

## Transformations and parameter regions

Claim derivation, deduplication, scoring, falsification, and parameter analysis
are separate mathematical operations with domain-owned contracts. Agents
compose them into generation, repair, or generalization strategies. Each
operation validates its inputs, stores material claims and edits as immutable
artifacts, and preserves lineage. Generated, repaired, and generalized
statements remain `UNVERIFIED`.

A parameter-region plugin may return `PROPOSED` or `SAMPLED` evidence only.
Jacobian commits an immutable `ParameterRegionSubject` binding the target claim,
region kind, exact conditions, and sample artifacts. Promotion requires an
authorized certificate record whose exact claim and subject artifact URIs are
parents of that record. The service replays the certificate and accepts the
promotion only if replay reproduces the same verification-record URI.
Mathematical interpretation of the region remains in the authorized checker;
the generic runtime only enforces bindings and evidence state.

## MCP boundary

The engine is also available as a Python library and CLI. The public MCP
surface implements [search and execute](#search-and-execute) and stays
deliberately small:

- `math.find` searches or inspects installed math tools (capabilities).
- `math.run` executes one named math tool by ID.
- `capability://catalog` exposes the full installed capability inventory.
- Resources expose large artifacts, traces, and experiment state.
- Tool responses return typed mathematical values in MCP structured content;
  durable or large objects are exposed as resource URIs.
- Long-running bounded searches return an experiment handle when their
  capability contract requires durable state.
- Scope and archive artifacts are immutable; the experiment snapshot is a
  durable lifecycle record.

The MCP Python SDK `2.0.0` owns statically declared tool schemas, typed result
serialization and validation, structured content, progress, and transport
cancellation. Jacobian retains descriptor-selected validation for the dynamic
`math.run` payload because the selected mathematical schema is not
known when the MCP tool is registered. Ordinary responses return the Pydantic
`CapabilityResult` directly so the SDK produces synchronized model-visible
`content` and application-facing `structured_content`. An explicit
`CallToolResult` is reserved for responses that genuinely require MCP content
blocks such as a `ResourceLink`, custom metadata, or a deliberate text
projection.

When a result needs durable retrieval, its typed result includes the canonical
artifact URI and the adapter may additionally emit an MCP `ResourceLink` with
the resource's JSON media type and known size. Tenant and resource
authorization follow the direct resource-read path.

The Python SDK advertises these stable tools and static resources through the
`io.jacobian/core` v1 extension. They remain ordinary MCP tools and resources:
clients do not need to advertise a matching client extension to invoke them.
Prompts and parameterized resource templates are registered through the SDK's
public server APIs because extensions intentionally do not contribute those
protocol objects. Adding a mathematical domain never adds another extension or
top-level MCP tool.

The engine does not expose a generic public `solver.solve`. Solver families have
different inputs, guarantees, and certificates, and remain typed internal
backends.
