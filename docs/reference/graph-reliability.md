# Small exact graph reliability

[Documentation home](../index.md)

- Capability: `probability.graph_reliability.connection_probability.compute`
- Verification: `probability.graph_reliability.connection_probability.verify`
- Producer: pinned Python-FLINT exact rational arithmetic
- Checker: isolated standard-library `Fraction` enumeration and graph traversal

The capability computes one atomic outcome: the exact probability that two
explicit terminals are connected when every edge of one finite simple
undirected graph is independently open with its declared rational probability.

## Bounded contract

The request uses the existing canonical `SimpleUndirectedGraph`, at most 16
vertices and 12 edges, one rational probability in \([0,1]\) for every edge in
the graph's exact order, and two distinct declared terminals. Full request
validation precedes computation and artifact writes.

The producer exhausts all \(2^{|E|}\) edge subsets. Every state records its
canonical index, open edges, exact product-measure mass, and terminal
connectivity. The result reports `visited_states`, `COMPLETE_EDGE_SUBSETS`,
`COMPLETE`, `truncated=false`, and `EXHAUSTED` separately from execution and
assurance. It therefore has no prefix, timeout, or failed-witness semantics.

The producer remains `COMPUTED`. The operator-authorized checker independently
parses the graph and probabilities, enumerates the entire powerset with
standard-library `Fraction`, recomputes connectivity by graph traversal, and
binds every state and the final sum to the stored request and result.

## Excluded claims

This capability does not construct bunkbed graphs, compare two events, exploit
symmetry, emit a reliability polynomial, prescribe a search, or support more
than 12 edges. In particular, the explicit 7,222-vertex counterexample in
[The bunkbed conjecture is false](https://arxiv.org/abs/2410.02545) is
answer-visible motivation, not an acceptance target. Scaling that case requires
separate compressed graph and symmetry artifacts; raising a JSON limit would
not establish a complete calculation.

The separately verified
[declared graph-symmetry orbit](graph-symmetry-orbits.md) capability exposes
vertex and edge compression metadata under supplied generators. It does not
lift that action to edge states or establish a reliability identity.

## Development handoff

### Discovery and implementation

- Decision: accept only bounded terminal connection probability.
- Portfolio delta: existing graph capabilities compute structural invariants;
  finite-distribution probability operations do not bind independent edge
  states to graph connectivity.
- Contract: 16 vertices, 12 edges, 4,096 states maximum; exact heterogeneous
  rational probabilities; complete state ledger.
- Base revision: `765cf5cb48dcf0f2b40e8dc3bffc3988ae8aa7e1`.
- Runtime: CPython 3.12 and pinned `python-flint==0.9.0`; no dependency change.
- Public reproduction:
  [`reliability-single-edge`](../../benchmarks/datasets/public-reproductions-v1/reliability-single-edge/).

### Checker and evaluation

- Exact claim: the reported sum is exactly the mass of every edge subset in
  which the stored terminals are connected.
- Independence: clean-process checker, no FLINT and no producer import.
- Attacks: changed connectivity, state mass, state omission, metadata, source,
  and fresh payload digest must produce `REJECTED/UNKNOWN` and no record.
- Public cases are answer-visible and unscored.
- Held-out control/treatment protocol is frozen as `READY_NOT_RUN`; no model,
  prompt, or raw trace exists and no portfolio improvement is claimed.
- Primary evaluation metrics are exact-answer correctness, checker-bound
  completion, false completeness, and false `VERIFIED` rate.
- Decision: keep the bounded experimental operation; hand compressed/symmetric
  scaling to a separate discovery batch.

### Reproducibility snapshot

- Installed catalog: 284 capabilities, catalog version 1,
  `sha256:532e6ccb09bc8552c5a10d7d464b211546f514b83c9f86db7543d4e46432ed8c`.
- Policy: `DEFAULT`,
  `sha256:870a92b83d3e522e4015b6bb1cabda33086906f9de1c3c36e466251ea7ed1957`.
- Producer runtime: `python-flint==0.9.0`, available on `linux-x86_64`;
  Python distribution RECORD digest
  `sha256:8ade9b4c5c1972b029d9393bb2586e2097cc44149a84ef8ef9ef376d634c328f`.
- Validation: focused producer, checker, verification, discovery, and public
  reproduction tests passed; `make check` passed with 288 unit tests; and
  `make test-affected BASE=origin/main` passed every selected lane (`unit`,
  `component`, `domain`, `composition`, `storage`, `process`, `mcp`,
  `provider`, `e2e`, `static`, `build`, and `docs`). Skips were limited to the
  documented unavailable Lean and external proof runtimes.
- Open obligations: the held-out model-in-the-loop comparison is not run, and
  no claim is made for compressed graphs, symmetry reduction, reliability
  polynomials, bunkbed event comparison, or the 7,222-vertex counterexample.
