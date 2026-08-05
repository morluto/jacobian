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

Search, generation, transformation, retrieval, and evaluation primitives may
return useful unverified results. They cannot promote their own output to
verified evidence. Promotion requires an independently authorized checker bound
to the exact claim, semantics, candidate, scope, certificate format, and
checker identity.

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

- A universal `solve_conjecture` endpoint
- Reimplementing Lean, Alloy, SAT/SMT, CAS, or optimization engines
- Treating a caller's self-review as independent verification
- Requiring verification for every computation or retrieval
- Letting a database entry become true because it is popular or highly ranked
- Claiming process isolation from a Python child process or bearer token alone
