# Capability surface

[Documentation home](../index.md)

- Status: Current pre-stable MCP surface
- Installed membership remains runtime-defined

Jacobian exposes mathematical operations as namespaced capabilities. The
model-facing MCP surface contains two capability tools and, in `REQUIRED` or
`AUDIT` mode, one operational reasoning-log tool:

| MCP tool | Purpose |
| --- | --- |
| `capability.describe` | Search the compact installed index, or read one capability's exact schemas by ID. |
| `capability.invoke` | Invoke an installed capability in `EXPLORE` or `VERIFY` mode. |
| `reasoning.write` | Append a bounded model-authored `PLAN`, `BEFORE_TOOL`, `AFTER_TOOL`, or `FINAL` summary. Available only in `REQUIRED` or `AUDIT` mode. |

The reasoning log is not a mathematical capability, proof object, workspace, or
chain-of-thought collector. In `REQUIRED` mode, every `capability.invoke` must
carry the `reasoning_run_id` and `reasoning_call_id` returned by the current
`BEFORE_TOOL`. The server binds the actual execution status, assurance,
completeness, result digest, and artifact URIs, then records whether the model's
structured `AFTER_TOOL` report matches them, without copying the capability
payload or output into the log. See the
[reasoning-log protocol](reasoning-log.md).

Read `capability://catalog` to discover installed capability IDs, provider
versions, supported modes, compact schemas, and tags. Catalog membership means
that an operation is installed and invocable. It does not imply compatibility
support, recommendation, conformance coverage, or authority to return
`VERIFIED`.

There are no alternate mathematical capability profiles and no public top-level
MCP commands for individual mathematical operations. The operator may set the
reasoning-log enforcement mode to `REQUIRED`, `AUDIT`, or `OFF`; `OFF` is the
default and preserves the legacy two-tool surface. Adding a capability does not
add a new MCP tool.

## Capability contract

Each capability has one agent-visible mathematical outcome. It consumes typed
inputs and returns a typed result with:

- execution status and operation-specific output;
- artifact references and relationships;
- scope and completeness;
- exact, approximate, bounded, exhaustive, deterministic, or heuristic
  qualifiers as applicable;
- assurance and any remaining proof obligations;
- provider and execution provenance.

Installed descriptors expose the exact provider version, digest kind and
digest, platform, install tier, license metadata, detected features, and fixed
checker identities. Results repeat the selected provider and provider digest.
The [provider runtime contract](provider-runtime.md) defines health probing,
fail-closed registration, and repeatable local measurement.

Backend-call atomicity is not required. An adapter may coordinate several
backend calls when they jointly implement one coherent operation, but it must
not hide mathematically useful intermediate values or artifacts, failures,
relationships, or obligations.

`EXPLORE` returns proposed, heuristic, or computed evidence. `VERIFY` may
return `VERIFIED` only when an operator-authorized independent checker accepts
evidence bound to the exact claim, semantics, candidate, scope, certificate
format, and checker version. Search, generation, evaluation, and computation
cannot certify their own conclusions.

Invalid requests, adapter failures, timeouts, and cancellations return
stage-aware diagnostics. They do not become mathematical conclusions.
Domain adapters validate their complete Pydantic request model before
computation or artifact writes. JSON Schema remains the discovery contract;
Pydantic enforces cross-field conditions such as polynomial-map dimensions,
finite operation-table closure, and bounded exact encodings.

## Installed capability discovery

The installed catalog is the canonical capability inventory. Its membership
depends on the available provider runtimes, operator-authorized checkers,
enabled bundled references, configured exclusions, and operator-installed
adapters. A static list in this document would therefore describe only one
installation snapshot.

`capability.describe` has two forms. Search or browse to retrieve compact
installed outcomes without loading every schema:

```json
{
  "query": "find a counterexample to associativity",
  "domain": "universal_algebra",
  "mode": "EXPLORE",
  "limit": 5
}
```

`query` searches published capability IDs, titles, descriptions, and tags.
`domain` filters the domain-owned capability namespace, with exact tag matches
also accepted. `mode` and `limit` are optional; `limit` defaults to 5 and is
bounded from 1 through 20. Omit `query` to browse the installed inventory in
stable ID order. Each response includes stable catalog and operator-policy
digests and is bounded to 16 KiB; when `next_cursor` is present, pass it back
with the same filters and limit to continue. Results report `matched_on` and
`matched_terms`; their deterministic ranking is retrieval, not a recommendation
or mathematical strategy. Query results also report a deterministic
`relevance_score`, `query_coverage_milli`, and per-match `lexical_fit`. The
result-level `portfolio_fit` distinguishes strong candidates, only weak lexical
matches, and no lexical matches. These are transparent descriptor-retrieval
signals, not a proof that an operation is mathematically suitable or absent.
In particular, top-N ordering among `WEAK_LEXICAL_MATCH` entries must not be
treated as capability fit. Start with five results, inspect only the strongest
one or two relevant contracts, then search again only when useful.

Discovery can also be constrained by `input_kind`. Installed descriptors
declare whether they accept a structured request, formal proposition, or typed
artifact. A typed artifact search also supplies the exact `schema_uri` from its
stored artifact manifest as `artifact_type`.
General natural-language proof prose is not a formal artifact: declaring
`NATURAL_LANGUAGE_PROOF`, or using an unambiguous phrase such as “informal
proof” or “proof prose,” returns typed `NO_ROUTE` unless an installed provider
explicitly accepts that input. The response's `routing_status` and
`routing_basis` are separate from lexical `portfolio_fit`.

Call `capability.describe` again with one returned `capability_id` to receive
the default `SUMMARY` exact projection. It is for judging fit and contains the
one-line outcome, modes, tags, provider availability, input/output field
summaries, and whether descriptor-owned invocation examples are available:

```json
{"capability_id": "universal_algebra.search.countermodel"}
```

Once the outcome fits, request `view: "CONTRACT"` before invoking. It adds the
complete validation-equivalent input schema (annotation/default and
discriminator routing metadata are omitted), concise output/runtime summaries,
related operations, and descriptor-owned validated invocation examples:

```json
{
  "capability_id": "universal_algebra.search.countermodel",
  "view": "CONTRACT"
}
```

Use `view: "FULL"` when complete output schema, provider
configuration, licensing, or other audit metadata is required:

```json
{
  "capability_id": "universal_algebra.search.countermodel",
  "view": "FULL"
}
```

Once a domain-owned producer fits the outcome, invoke it before separately
searching for a checker. Follow the checker, certificate, and verification
fields in the producer result rather than guessing that a generic verifier
accepts its artifact.

`capability.invoke` returns the Pydantic `CapabilityResult`. MCP Python SDK 2.0
derives its output schema, validates the returned value, serializes
model-visible `content`, and supplies the same typed value in
`structured_content`. Small, bounded mathematical outputs remain inline in the
result. An empty `artifact_uris` means the value was not retained.

Capabilities return resource URIs only when their mathematical outcome needs
durable identity, independent retrieval, replay, resumability, evidence
binding, or size-separated transport. An adapter uses an explicit
`CallToolResult` only when it must add an MCP content block such as
`ResourceLink`, custom metadata, or a deliberate text projection.

Published invocation examples are validated against the descriptor schema when
the capability is installed. Domain-owned examples may additionally be
constructed through the complete Pydantic request model. They illustrate valid
calls; they do not prescribe a research workflow.

Every exact projection includes a scope rule. An invocation covers only its
exact supplied input or claim. Additional finite or bounded invocations remain
finite evidence and do not establish an all-orders, all-parameters, or
otherwise unbounded conclusion. A supplied claim may itself be universal—for
example, a formally checked theorem—but bounded examples do not silently widen
their own scope.

Read `capability://catalog` when a client or operator needs the complete
machine-readable inventory in one response. Do not infer current installation
membership or payload fields from static documentation.

The [domain operation library](domain-operation-library.md) defines the shared
contract for built-in mathematical operations. Capability-specific artifact,
provider, and verification references live with their owning domain. They are
intentionally not registered here: installing or documenting a capability
does not change the generic MCP tool contract.

## Mathematical operation portfolio

The portfolio may include capabilities for operations such as:

- artifact materialization;
- claim validation;
- candidate evaluation;
- witness search and independent witness checking;
- certificate replay;
- bounded enumeration and canonicalization;
- exact invariant computation;
- representation and claim transformation;
- finite-family materialization;
- premise and research-record retrieval;
- proof-assistant checking;
- exact separation, constraint solving, or construction.

These are capability families, not a required taxonomy. Use domain-specific
IDs and contracts where mathematical semantics differ. For example,
`graph.enumerate.nonisomorphic` and
`polynomial.compute.groebner_basis` should not be forced through a universal
object or solver schema.

`graph.construct.explicit` validates a complete bounded vertex/edge request
before writing anything, canonicalizes labels and undirected edges, and returns
the domain-owned simple-graph artifact accepted by graph consumers. Generic
`artifact.put` still does not authorize graph semantics.
`graph.induced_tree.maximum.verify` independently exhausts all vertex subsets
for stored exact producer results of order at most 14. It binds the complete
graph-optimization input and result lineage and does not reuse the producer's
Z3 search. Larger inputs return an unsupported non-conclusion.

Useful low-level operations may retain descriptive IDs such as
`claim.validate`, `witness.find`, `witness.verify`, or
`certificate.verify`. Those names identify capabilities invoked through
`capability.invoke`; they are not separate MCP tools.

`claim.conjunction.split` and `claim.implication.obligations` operate on the
registered v1 `PROPOSITIONAL_STRUCTURE` artifact. They return only immediate,
ordered subtrees plus source-bound reconstruction data, preserve nested
grouping, and report `COMPUTED` rather than proof verification. Raw natural
language and printed Lean expressions are outside this contract; exact Lean
decomposition requires a future typed elaborated-expression artifact.

Opaque multi-stage commands are not part of the public surface. Agents should
compose generation, evaluation, ranking, falsification,
refinement, and verification from separately invocable capabilities. An
optional workflow capability is appropriate only when it has one coherent
mathematical outcome and preserves visible intermediate values, artifacts, and
assurance boundaries.

## Adapters and trust boundaries

Capability adapters connect maintained proof assistants, CAS systems, solvers,
mathematical databases, and domain libraries to the common contract. Domain
plugins own mathematical schemas, transformations, invariant meanings, and
required checker roles. The runtime owns artifact identity, budgets, execution
status, provenance, assurance, and checker authorization.

SAT, SMT, LP, MIP, SyGuS, interval arithmetic, and proof assistants should use
typed domain adapters with explicit certificate formats. Jacobian does not
expose a generic `solver.solve` or `sandbox.run` truth primitive.

An adapter or plugin cannot authorize its own checker. Checker administration
is operator-controlled and outside the model-facing MCP surface.

Operators may additionally constrain visible and invocable capabilities by
exact ID, domain, tag, or mode. The
`COMPUTE_VERIFY_NO_RETRIEVAL` profile denies retrieval-tagged capabilities; it
is intended for evaluation isolation where only computation and independent
verification should be available. Catalog and
discovery responses bind the active policy profile and digest. A direct call to
a hidden capability fails with `CAPABILITY_POLICY_DENIED`. Capability policy
changes availability only: it cannot install a checker, authorize one, or
change verification authority.

## Operating guidance and prompts

The initialization response describes when the mathematical toolbox may help
and points to the two-tool discovery interface. It does not choose task
decomposition, proof strategy, capability composition, iteration, or stopping
criteria.

Read `jacobian://instructions` to recover the complete operating model without
reconnecting. The resource explains discovery, exact contract inspection,
composition, result dimensions, verification boundaries, and artifacts.

Two optional MCP prompts provide protocol scaffolding:

- `jacobian-discover` turns a mathematical task into discovery and exact-contract
  steps while leaving strategy with the agent.
- `jacobian-check-evidence` explains how to look for a compatible independent
  checker without treating search or computed evidence as verified.

Clients that do not support resources or prompts can use the same tools from
their published descriptions and schemas.

## Resources

Read-only discovery and large-object access use MCP resources:

```text
jacobian://instructions
artifact://sha256/<digest>
capability://catalog
reference://catalog
experiment://<id>
experiment://<id>/accounting
experiment://<id>/scope
experiment://<id>/archive
```

Only resource templates implemented by the installed runtime are advertised.
Schemas, semantics, plugin manifests, witnesses, certificates, and
verification records are ordinary artifacts. Resource access does not alter
their assurance.
