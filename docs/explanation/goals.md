# Product goals

[Documentation home](../index.md)

- Status: Active direction
- Planning model: Rolling goals, pursued in parallel
- Related records: [Product model](product-blueprint.md) and
  [architecture](architecture.md)

Jacobian is building a broad toolbox of composable mathematical capabilities
for AI agents. The project does not follow a linear sequence in which search,
claim transformation, retrieval, or formal proof must mature before another
area can be explored. Useful capabilities may be added, evaluated, revised, or
removed independently.

Current reference documents and the installed catalog describe supported
behavior. They do not prescribe research order, block experimental
capabilities, or promise that an idea will become a stable interface.

## Rolling goals

### Expand mathematical capability

Expose useful operations from maintained proof assistants, computer algebra
systems, solvers, optimization systems, databases, and domain libraries.
Prefer capabilities with one clear, agent-visible mathematical outcome.
Experimental adapters may change without compatibility guarantees.

### Improve agent discovery and composition

Help agents find relevant capabilities without loading every schema into their
initial context. Improve descriptors, examples, catalog search, ranking, and
value and artifact relationships from observed agent behavior. Keep research
strategy with the agent rather than encoding one preferred proof workflow in
the runtime.

### Increase independent verification coverage

Add independent checkers for exact claims and evidence where they close real
trust gaps. Bind verification to the exact claim, semantics, candidate, scope,
certificate format, and checker identity. Computation, search, retrieval, and
model judgment never verify themselves.

### Evaluate portfolios on mathematical work

Use held-out datasets, hidden oracles, real transcripts, and paired ablations
to measure complete capability portfolios as well as individual operations.
Track correctness, false certification, runtime, tokens, tool calls, and
parameter errors. Use the evidence to improve discovery, defaults,
consolidation, and retirement—not to gate experimentation.

### Preserve transparent mathematical work

Keep important intermediate objects, failed attempts, transformations,
relationships, and proof obligations inspectable. Retain them durably when
identity, replay, resumability, evidence binding, or size requires it. Optional
workflows may coordinate several operations, but must not erase their evidence
or independent verification boundaries.

### Keep the public surface small

Expose mathematical breadth through namespaced capabilities behind
`math.find` and `math.run`, not through a growing set of
top-level MCP tools. Remove compatibility wrappers and duplicate interfaces
when they no longer serve a supported contract. Reuse maintained mathematical
backends rather than accumulating custom infrastructure.

## How priorities change

Work may advance on any goal when a dataset, transcript, backend, or trust gap
provides a useful test. Concrete implementation work belongs in focused GitHub
issues with observable success criteria. Current reference documents record
supported contracts; benchmarks and evaluation reports record evidence.

The direction is working when better agents can discover and compose the same
portfolio more effectively, exact conclusions remain independently
checkable, and adding a mathematical backend does not require redesigning the
runtime or MCP surface.
