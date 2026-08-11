# Product model: mathematical tools for AI agents

[Documentation home](../index.md)

- Status: Active product direction
- Scope: Mathematical tools, agent workflows, capability adapters, research
  memory, and optional mathematical assurance

## Product definition

Jacobian is an MCP server, CLI, and Python library that exposes a toolbox of
composable mathematical capabilities to AI agents. Its goal is to help agents
and human researchers make trustworthy progress on conjectures and other
problems that benefit from executable search and checkable evidence.

The product direction is:

- broad portfolio of mathematical capabilities;
- mathematically atomic, agent-visible outcomes;
- agent-owned composition and research strategy;
- optional workflows with inspectable intermediate values and durable artifacts;
- independent verification of exact claims and evidence.

Each capability performs one observable mathematical operation and returns
typed, inspectable results with explicit relationships, scope, execution
status, assurance, and provenance. Existing mathematical software and domain
plugins supply the mathematics; capability adapters expose it through a common
contract. Jacobian supplies operations, optional durable artifacts, execution
policy, and trust boundaries—not a prescribed research strategy.

### Why atomic capabilities scale

An atomic capability has one clear, inspectable mathematical outcome: factor a
polynomial, check a proof, or find paths in a graph—not “solve the whole
problem.” This makes each capability easier to test and trust because its
inputs, outputs, resource bounds, and evidence are explicit. Agents can
combine those capabilities into new strategies without waiting for a custom
workflow for every problem, while future models remain free to choose different
decompositions.

Atomic outcomes also keep failures useful. A result can show that a factorization
failed, a search timed out, or a proof checker rejected a claim instead of
collapsing everything into an opaque solver failure. The portfolio can grow one
reusable capability at a time, while agents—not the runtime—own the strategy
that composes them.

The product is the mathematical toolset and its shared capability runtime:
versioned contracts, typed value transport, artifact and provenance storage,
execution and budget control, adapter and plugin boundaries, and optional
checker-backed assurance.
MCP is the primary agent-facing interface; the CLI and Python API support local
use and integration without changing the mathematical contracts.

The product succeeds when an agent solves held-out tasks more reliably or
efficiently with Jacobian than the same agent solves them with prompts and a
general-purpose shell alone. Starting an MCP server, calling a tool, or
producing a verification record is necessary infrastructure evidence, not
proof of that product outcome.

## Why use Jacobian instead of asking the agent to do the math?

Agents retain useful mathematical knowledge and should own the research work:
choosing a representation, proposing a theorem or candidate, deciding which
tool to try, and interpreting the result. Jacobian does not replace that work.
It makes the externally executed part of an investigation explicit when exact
calculation, finite search, solver output, or formal proof checking matters.

For example, a search can return a **witness**: a concrete object that
establishes an existential claim, or a counterexample that disproves a
universal one. The search result is useful evidence, but it is not itself a
verified conclusion. An independent checker must validate that exact object
against the exact claim and declared scope. A completed search with no witness
is likewise not a conclusion unless the scope is complete and that
completeness is established.

The distinction is deliberately visible in the result contract:

- `HEURISTIC` covers results that depend on an unchecked witness, model,
  sampling, or untrusted search;
- `COMPUTED` covers deterministic results whose software contract is tested;
- `VERIFIED` requires independently checked evidence bound to the claim,
  candidate, semantics, scope, certificate format, and checker identity.

Lean, SAT proof checkers, CAS systems, and solvers are complementary backends,
not alternatives to Jacobian's product boundary. A Lean kernel checks a formal
proof term; a SAT certificate checker replays a finite Boolean proof; a CAS or
solver may calculate or search. Jacobian gives an agent one way to discover,
invoke, compose, and retain the results of those systems without mistaking a
backend's successful run for independent verification.

The division of responsibility is compact:

| System | Main job |
| --- | --- |
| Model | Proposes ideas and chooses a mathematical strategy |
| CAS / SAT / SMT | Calculates or searches in a specialized domain |
| Lean | Checks a formal, general mathematical proof |
| Jacobian | Exposes supported backends and domain operations to agents with typed results, scope, evidence, provenance, and verification status |

## Tool and primitive contract

At the product level these capabilities are tools. Internally, the target
mathematical primitive contract is a versioned capability with one observable,
agent-visible outcome. Backend-call atomicity is not required: an adapter may
coordinate several backend calls when they jointly produce that outcome. A
capability consumes typed values and artifact references as appropriate and
returns:

- a typed inline output, or a reference to a durable mathematical object;
- explicit relationships to its inputs when durable objects are involved;
- any proof obligations created by the operation;
- execution status and resource accounting;
- assurance and the evidence supporting it;
- enough provenance to replay or compare the step.

Composition uses three boundaries:

```text
typed values compose computations
artifact references compose durable evidence
verification records establish trust
```

Shared mathematical objects are broad, canonical values, not promises that
every operation will execute. A matrix, polynomial, or finite complex can be
valid at its object boundary yet exceed a particular operation's preflight
budget. Shape, exact scalar representation, ordering, and normalization belong
to the object; compatible dimensions, coefficient-growth estimates, dense
construction costs, and algorithm-specific limits belong to the request that
uses it.

Small bounded results stay inline and can be inserted directly into a later
typed request. Do not materialize an inline value merely to hand it to another
ordinary operation. Artifacts are explicit for durable identity, raw proof or
certificate files, resumable state, independent retrieval, or output whose
complete inline form would be unusable. A result that exceeds its declared
representation envelope fails explicitly rather than being truncated,
approximated, or silently spilled to storage.

Search, generation, transformation, retrieval, and evaluation primitives may
return useful unverified results. They cannot promote their own output to
verified evidence. Promotion requires an independently authorized checker bound
to the exact claim, semantics, candidate, scope, certificate format, and
checker identity.

For an inline exact replay, the checker receives the authoritative input and
candidate directly; neither is first materialized as an artifact. An accepted
replay persists one verification record, whose parent is the immutable
semantics artifact. The result exposes both URIs in `artifact_uris`: the record
is the decision evidence and the semantics artifact is the contract it binds.
Rejected, malformed, cancelled, timed-out, or incomplete replays create no
verification record and establish no mathematical conclusion.

Broad actions such as “investigate this conjecture” are workflows, not
primitives. A workflow may coordinate many primitive calls, but it must expose
useful stage values and artifacts and preserve their separate assurance labels.
Jacobian does not retain parallel top-level MCP tools or compatibility façades
for these workflows; agents compose the namespaced capabilities.

Design new capabilities against the installed portfolio. When an existing
typed value or artifact already exposes the needed mathematical outcome,
consume it instead of silently recomputing or materializing it. Temporary
overlap is acceptable
for experimentation, performance, batching, backend constraints, or a
genuinely different agent-visible outcome. State the overlap explicitly and
preserve useful intermediate values and artifacts. If two capabilities expose
the same outcome, consider discovery, contract clarity, artifact handoff,
consolidation, or retirement before adding another stable ID.

## Ownership model

The boundaries are intentionally narrow:

- The runtime owns artifact identity, execution status, assurance, checker
  authorization, budgets, and provenance.
- The MCP SDK owns static tool schemas, typed result serialization and
  validation, structured content, progress, and transport cancellation.
- Capability adapters connect external SAT, SMT, CAS, optimization, retrieval,
  and proof systems to the primitive contract.
- Domain plugins own mathematical schemas, transformations, invariants,
  witness meanings, and required checker roles.
- Independent checker packages implement replay; operators authorize them.
- Agents own multi-step exploration and proof strategies.
- Reference scenarios and benchmarks own worked examples.

This separation lets a new mathematical operation or external engine appear
behind a capability ID without changing the MCP server or expanding checker
authority.

## System shape

```text
Codex CLI              ChatGPT / remote agent
    │ STDIO                    │ Streamable HTTP
    └──────────────┬───────────┘
                   ▼
               MCP projection
                       │
                       ▼
          capability://catalog
          math.find
          math.run
                       │
                       ▼
               CapabilityService
       ├─ knowledge and memory
       ├─ math/solver adapters
       │  Lean SAT SMT CAS Alloy domain tools
       └─ experiment services
       │
       │ optional promotion
       ▼
        authorized checker / proof engine
                   │
                   ▼
          immutable verification record
```

The generic capability layer must understand an operation ID, JSON schemas,
supported modes, execution status, assurance, scope, artifact relationships,
proof obligations, and an episode handle. Mathematical semantics remain in
adapters and domain plugins.

## Capability contract

An adapter registers one `CapabilityDescriptor` and implements:

```python
class CapabilityAdapter(Protocol):
    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...
```

The descriptor declares:

- namespaced capability ID and adapter contract version;
- provider and concise model-facing description;
- supported `EXPLORE` and `VERIFY` modes;
- closed input and output JSON Schemas;
- read-only and episode-recording behavior;
- discovery tags.

`CapabilityService` validates both sides of the call, enforces identity and
mode, checks any verified record and its complete parent binding against the
local artifact store, and records the episode. Projected record IDs and
conclusions must agree with the checked record. `MCPServer` does not need a new
tool when an Alloy, Lean, SAT/SMT, CAS, or domain adapter is registered.

`CapabilityResult` version 2 exposes a generic operation-specific output plus
first-class scope, completeness, relationships, proof obligations, and artifact
URIs. The shared layer validates artifact bindings and checker-backed lifecycle
states; domain adapters still define the mathematical meaning of relation IDs,
scope parameters, and obligation artifacts.

The [capability workflow evaluation plan](../reference/evaluations/benchmark-contracts.md)
defines held-out workflows used to evaluate discovery, routing, defaults,
consolidation, and retirement. Experimental and version-breaking adapters may
be exposed before those evaluations show lift; prescribed-tool cases test
contract conformance, not autonomous portfolio value.

Deploy an operator-approved adapter package with a repeatable
`--capability-adapter package.module:factory` option. The factory receives the
tenant's explicit `ApplicationServices` graph and returns a
`CapabilityAdapter`. Loading Python code is an operator action, never a model
tool; it establishes availability, not mathematical trust.

The always-available bundled catalog contains:

- `artifact.put` for storing a typed mathematical object as an immutable
  artifact;
- `claim.validate` for checking a claim against its declared domain semantics;
- `evaluate.batch` for applying an evaluator to an explicit candidate batch;
- `witness.find` for bounded witness search;
- `witness.verify` for independently replaying witness evidence;
- `certificate.verify` for independently replaying certificate evidence;
- `case.partition.finite` for finite partition construction and independent
  coverage replay when an operator authorizes the bundled checker;
- `graph.search.atlas` for bounded exact-order construction from NetworkX's
  maintained Graph Atlas;
- `graph.compute.properties` for exact batched properties over Jacobian graph
  artifacts.

When the operator enables bundled references, the catalog also contains
`lean.check` for checker-backed Lean proof replay. These capabilities are not a
closed ontology.

A reference-domain investigation composes the atomic operations rather than
calling an opaque solver workflow: store the object with `artifact.put`,
validate the claim with `claim.validate`, evaluate candidates with
`evaluate.batch`, search with `witness.find`, and independently replay the
result with `witness.verify` or `certificate.verify`.

## Two lanes

### Explore

`EXPLORE` is the default. It may evaluate, search, query memory, call solvers,
generate candidates, or find witnesses. It does not run a checker merely to
make an intermediate result usable.

Explore results use:

- `HEURISTIC` when the result depends on search, an untrusted plugin, a model,
  sampling, or an unchecked witness;
- `COMPUTED` for a deterministic operation whose software contract is tested
  but which does not establish a promoted mathematical claim.

### Verify

`VERIFY` is requested for a durable theorem, counterexample, equivalence,
optimality, exhaustive scope, or reusable database fact. The adapter may invoke
an authorized checker or formal proof engine.

An adapter cannot create verified authority. `CapabilityService` accepts
`VERIFIED` only when the result names a valid local verification-record
artifact and exposes its checked evidence. Failure falls closed to a
non-verified result or operational error.

## Local and remote hosts

The local Codex host uses STDIO. It exposes `capability://catalog`,
`math.find`, and `math.run` rather than projecting backend
mathematical operations or workflows as additional top-level MCP tools.

Remote hosts use Streamable HTTP and subject-bound tenant state. Authentication,
tenant isolation, persistence, and TLS are deployment responsibilities, not
mathematical primitives. See
[Deploy the remote MCP server](../how-to/deploy-remote-mcp.md) for their
concrete requirements.

## Product evidence

The immediate product work is to stabilize the primitive contract, make stage
composition visible, and exercise external adapters without runtime or MCP
edits. Authenticated hosting and compact tool projection support that work;
they are not substitutes for useful mathematical operations.

Agent evaluations measure held-out mathematical tasks, including counterexample
search, claim transformation, proof decomposition, premise retrieval, and
independent replay. Their role is to improve discovery, routing, defaults,
consolidation, and retirement of capabilities, not to gate exposure of
experimental adapters. Prescribed-tool cases test contract conformance; only
agent-chosen-tool cases measure autonomous portfolio value. Cross-project
corpus providers follow only after local episode queries are empirically
useful. The [agent evaluation protocol](../reference/evaluations/evaluation-methods.md)
defines the controls, retained evidence, and scoring required for a product
claim.

## Non-goals

- A universal mathematical ontology
- A natural-language-to-formal-mathematics translator
- Distributed search infrastructure
- An opaque generic solver in the kernel
- A universal `solve_conjecture` endpoint
- Reimplementing theorem provers or SAT/MIP solvers
- Reimplementing Lean, Alloy, SAT/SMT, CAS, or optimization engines
- Accepting arbitrary model-supplied executable bundles
- Treating floating-point scores, timeouts, and solver labels as proofs
- Treating a caller's self-review as independent verification
- Requiring verification for every computation or retrieval
- Letting a database entry become true because it is popular or highly ranked
- Claiming process isolation from a Python child process or bearer token alone
