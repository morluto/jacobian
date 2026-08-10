# Product model: mathematical tools for AI agents

[Documentation home](../index.md)

- Status: Active product direction
- Scope: Math tools for agents, search/execute surface, adapters, optional
  durable artifacts and independent checker tools

## Product definition

Jacobian is an MCP server, CLI, and Python library that gives AI agents a
**toolbox of atomic math tools**. Agents **find** tools and **run** them; they
compose results to investigate conjectures. The runtime does not own research
strategy.

```text
math.find   → search / inspect math tools
math.run    → execute one named tool → mathematical result
```

See [Search and execute](architecture.md#search-and-execute). Catalog entries
are still often called *capabilities* in the API; agent-facing prose prefers
**math tool** or **operation**.

The product direction is:

- a large portfolio of small, composable math tools;
- a **fixed** agent surface (`math.find` / `math.run`) as that portfolio grows;
- **math-first results**: ordinary tools return calculations (values), not
  trust slogans;
- agent-owned composition;
- optional durable artifacts when identity, size, or replay needs them;
- optional **checker tools as additional catalog IDs**—never dual-mode tools.

Existing CAS, SAT/SMT, Lean, solvers, and domain libraries supply the
mathematics. Jacobian exposes them under typed contracts with resource bounds
and clear run status. It is not a workflow engine and not a prescribed
explore/verify research process.

### Why atomic tools scale

An atomic tool has one clear mathematical outcome: factor a polynomial, take a
determinant, find a path, check a certificate—not “solve the whole problem.”
Inputs, outputs, and bounds stay testable. Agents combine tools into new
strategies without a custom workflow per paper. Failures stay useful
(timeout, invalid input, checker rejection) instead of one opaque solver
error.

The product succeeds when an agent does better mathematical work with Jacobian
than with prompts and a general-purpose shell alone. Starting a server or
emitting a verification record is infrastructure, not product proof.

## Why use Jacobian instead of asking the agent to do the math?

Agents keep mathematical judgment: representation, which tools to try, how to
compose results, when to stop. Jacobian runs the parts that benefit from exact
calculation, finite search, solvers, or formal checkers—and returns those
outputs as first-class values the agent can feed into the next tool.

Example:

```text
math.run(polynomial.compute.gcd, {left, right})  →  gcd polynomial
math.run(polynomial.gcd.verify, {input, candidate})  →  check verdict
  (only if the agent wants independent replay; separate tool ID)
```

A successful compute is a **value**. It is not automatically a formal theorem.
If independent checking matters, the agent runs a **checker tool**—another
entry in the same catalog—not a second mode on the producer.

Lean, SAT checkers, CAS, and solvers remain backends. Jacobian’s job is one
discoverable, composable portfolio in front of them.

| System | Main job |
| --- | --- |
| Model | Strategy: what to try and how to compose |
| CAS / SAT / SMT | Calculate or search in a specialized domain |
| Lean / checkers | Check formal or certificate claims when invoked |
| Jacobian | Search/execute surface over installed math tools and typed results |

## Tool contract

Each catalog entry is one tool: one ID, one mathematical role, one request
schema, one primary result shape.

**Ordinary (producer) tools** return:

- the **mathematical value** (inline or as an artifact reference when large /
  durable identity is required);
- **execution status** (completed, invalid input, timeout, error, cancelled);
- optional diagnostics and operator provenance (provider version, digests).

**Checker tools** are additional tools. They take a claim and candidate (or
certificate), run an operator-authorized independent checker, and return a
**verdict** (accepted / rejected / non-conclusion) plus bindings. They do not
share an ID with the producer.

Rules:

- **No dual-mode tools.** One ID is not both “compute X” and “verify X.”
- **No explore/verify research modes** as product design. Legacy wire `mode`
  fields are scheduled for removal
  ([#1143](https://github.com/morluto/jacobian/issues/1143)).
- **Failed or incomplete runs are not mathematical conclusions.**
- **Producers do not self-certify** as independently verified theorems.

Composition:

```text
typed values  →  feed the next math.run
artifacts     →  when durable identity or size requires it
checker tools →  optional independent check of specific claims
```

Shared mathematical objects (matrices, polynomials, graphs, …) are canonical
values. Object validity is not a promise that every tool will accept them under
every budget. Small results stay inline; do not materialize only to pass
between ordinary tools.

“Investigate this conjecture” is an **agent workflow** over many tool calls,
not one primitive. Jacobian does not add top-level MCP tools for workflows.

Design new tools against the installed portfolio. Reuse values and artifacts
that already expose the needed outcome. Prefer primitives that compose over
paper-shaped mega-tools.

## Ownership model

- **Agents** own multi-step strategy: which tools to find and run, how to
  compose values, when to invoke checker tools, when to stop.
- **Runtime** owns execution bounds, catalog install, artifact identity where
  used, and operator authorization of checker packages.
- **MCP SDK** owns static tool schemas, structured content, progress, and
  transport for the fixed `math.find` / `math.run` surface.
- **Domain packages** own mathematical schemas and kernels for ordinary tools.
- **Checker packages** implement independent replay; operators authorize them.
  Authorization is never self-granted by a producer.
- **Scenarios and benchmarks** own worked examples and evaluation evidence.

A new math tool or backend appears as a catalog ID, not as a new MCP tool.

## System shape

```text
Agent host
    │
    ▼
math.find / math.run / capability://catalog
    │
    ▼
Catalog of math tools
    ├─ ordinary tools  →  values (compute, search, transform, …)
    └─ checker tools   →  verdicts (optional; separate IDs)
         │
         ▼
  domain kernels / CAS / SAT / Lean / solvers
```

There is no product “promotion lane” the agent switches into. Checking is
running another tool.

## Capability contract (implementation name)

Adapters still register a `CapabilityDescriptor` and implement `invoke`. That
is the internal name for a math tool. The descriptor declares ID, version,
description, input/output schemas, and discovery metadata. New tools must
encode a **single** role (ordinary compute/search **or** checker)—not both.

`CapabilityService` validates requests, dispatches the tool, and enforces that
only authorized checker runs can claim independent acceptance with a bound
verification record. Ordinary tools are not required to narrate
HEURISTIC/COMPUTED/VERIFIED as their primary outcome; the primary outcome is
the mathematical value (or checker verdict). Legacy envelope fields may remain
on the wire during migration ([#1143](https://github.com/morluto/jacobian/issues/1143)).

Deploy operator-approved adapters via package entrypoints. Loading code is an
operator action, never a model tool; it establishes availability, not
mathematical trust.

Illustrative catalog entries (not a closed ontology):

- ordinary: `artifact.put`, `evaluate.batch`, `witness.find`,
  `graph.compute.properties`, `polynomial.compute.gcd`, …
- checkers (separate IDs): `witness.verify`, `certificate.verify`,
  `polynomial.gcd.verify`, `lean.check`, …

Example composition: put or compute values → search → optionally run
`witness.verify` on a found witness. Two tools, not one dual-mode tool.

Dual-mode leftovers (one ID advertising both compute and verify) are tech debt
to split; do not add more.

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

Product work is a useful math-tool portfolio behind search/execute: correct
values, composable contracts, and optional checker tools where they matter.
Hosting and wire format support that; they are not the product.

Agent evaluations measure held-out mathematical tasks (counterexamples,
transforms, premise retrieval, optional independent check). They improve
discovery, defaults, consolidation, and retirement—not gate every experimental
tool. Prescribed-tool cases test contracts; agent-chosen-tool cases measure
portfolio value. See
[agent evaluation protocol](../reference/evaluations/evaluation-methods.md).

## Non-goals

- A universal mathematical ontology
- A natural-language-to-formal-mathematics translator
- Dual-mode tools or explore/verify research phases as product design
- Assurance stamps as the primary agent-facing result of ordinary compute
- Distributed search infrastructure
- An opaque generic solver in the kernel
- One MCP tool per mathematical operation
- A universal `solve_conjecture` endpoint
- Reimplementing theorem provers or SAT/MIP solvers
- Reimplementing Lean, Alloy, SAT/SMT, CAS, or optimization engines
- Accepting arbitrary model-supplied executable bundles
- Treating floating-point scores, timeouts, and solver labels as proofs
- Treating a caller's self-review as independent verification
- Requiring verification for every computation or retrieval
- Letting a database entry become true because it is popular or highly ranked
- Claiming process isolation from a Python child process or bearer token alone
