# Four-domain capability roadmap

This reference freezes the discovery-to-implementation boundary for the
probability, discrete-mathematics, computational-geometry, and topology
foundations accepted on 2026-07-29. The machine-readable source of the catalog
snapshot, move ledger, case status, and discovery handoffs is
[`public_postdoc_status_v2.json`](../../benchmarks/research_challenges/public_postdoc_status_v2.json).

The source suite remains immutable and answer-visible. These records support
contract design and public regression; they are not a held-out model
evaluation. Each proposed producer remains capped at `COMPUTED`. A separately
authorized independent checker is required for `VERIFIED`.

## Frozen portfolio boundary

The pre-manifest environment at commit
`081834c979ec8f1c3b3995ebb86908bd82333a07` contains 269 capabilities under the
default policy and 266 under `COMPUTE_VERIFY_NO_RETRIEVAL`.

| Area | Counted prefixes | Installed count | Current boundary | Accepted foundation |
| --- | --- | ---: | --- | --- |
| Probability | `probability` | 1 | One finite rational raw moment | Event probability, conditioning, pushforward, convolution |
| Discrete mathematics | `combinatorics`, `finite_set`, `graph`, `lattice`, `sequence` | 95 | Broad graph/sequence/set computations; no finite-poset artifact | Poset materialization, width, linear-extension count, Möbius function |
| Computational geometry | `geometry`, `polytope` | 17 | Exact rational point/line/triangle/hull operations; no segment overlap or polygon membership | Segment intersection, polygon simplicity, point classification |
| Topology | `topology` | 0 | No topology-owned capability | Simplicial-complex materialization, chain complex, homology over a prime field |

Counts are snapshot facts, not rolling documentation. A later portfolio change
must create a new status overlay rather than edit this historical boundary.

## Shared contract rules

All four families must:

- expose one coherent mathematical outcome per capability;
- validate the complete request before computation or artifact writes;
- bind inline or artifact inputs under explicit exactly-one semantics;
- record scope, completeness, execution, conclusion, and assurance separately;
- preserve useful intermediate artifacts and relationships;
- include every first-class artifact reference in `artifact_uris`;
- make timeout, cancellation, provider errors, incomplete enumeration, and
  missing witnesses non-conclusions;
- keep optional-provider absence local to affected capability IDs; and
- use a producer-independent checker before returning `VERIFIED`.

## Finite rational probability

Accepted IDs:

- `probability.finite_distribution.event_probability.compute`
- `probability.finite_distribution.condition.compute`
- `probability.finite_distribution.pushforward.compute`
- `probability.finite_distribution.convolution.compute`

Shared input artifact:

```text
FiniteRationalDistribution
  canonical distinct rational atoms
  exact nonnegative rational masses
  exact normalization equal to one
  semantic version and content digest
```

The event contract takes an explicit finite atom selection, not executable
predicate code. Pushforward takes a total finite lookup map. Convolution states
independence as operation semantics and exposes pair contributions. A
zero-mass conditioning event returns a domain non-applicability conclusion
rather than an empty distribution or division error.

The independent checker uses standard-library rational arithmetic and binds
the source distribution, event or map, normalization, result, contribution
ledger, scope, and candidate digest. Gaussian all-parameter identities, large
graph reliability, finite Markov chains, and interval probability remain
separate gates.

## Finite posets

Accepted IDs:

- `poset.finite.materialize`
- `poset.width.compute`
- `poset.linear_extensions.count`
- `poset.mobius_function.compute`

The materialization request declares whether its relation contains cover edges
or comparable pairs. It validates labels, endpoints, acyclicity, antisymmetry,
and reflexive policy before writing artifacts. The result exposes a canonical
carrier, transitive closure, and Hasse reduction.

Width returns both a maximum antichain and a chain cover of the same size.
Those witnesses independently establish the lower and upper bounds. Linear
extension counting is bounded subset/ideal dynamic programming with explicit
state completeness. Möbius results declare whether they cover selected
intervals or a complete bounded matrix.

NetworkX may produce DAG and matching results. The checker independently
replays closure, chain and antichain predicates, dynamic-programming
recurrences, and Möbius convolution identities.

## Exact planar geometry

Accepted IDs:

- `geometry.segments.intersection.compute`
- `geometry.polygon.simple.decide`
- `geometry.polygon.point.classify`

Segment intersection returns a discriminated outcome:

```text
DISJOINT
POINT(point, PROPER | ENDPOINT_TOUCH | DEGENERATE_TOUCH)
OVERLAP(canonical maximal closed segment)
```

Polygon validity fixes repeated-closing-vertex, zero-edge, and self-touching
policies in the request semantics. Point classification returns exactly one of
`INSIDE`, `BOUNDARY`, or `OUTSIDE`; it does not run on an invalid polygon.

The producer may use the current pinned SymPy exact-rational backend. The
checker uses an independent rational determinant and interval-containment
implementation and checks every required nonadjacent edge pair for a positive
simplicity conclusion.

Exact Delaunay/Voronoi and rational H/V polyhedral conversion remain separate
CGAL and cddlib provider spikes.

## Finite simplicial topology

Accepted IDs:

- `topology.simplicial_complex.materialize`
- `topology.simplicial_complex.chain_complex.compute`
- `topology.simplicial_homology.compute`

The finite-complex contract fixes canonical vertex order, maximal-simplex
input, face closure, orientation, empty-simplex handling, isolated vertices,
and reduced versus unreduced conventions. Closure size and dimension are
bounded before artifact writes.

The chain-complex result exposes simplex bases and oriented boundary matrices.
The first homology contract uses one explicitly bounded prime field and
returns ranks, cycles, boundaries, and quotient-basis evidence.

The independent checker reconstructs every face and boundary, verifies
`boundary * boundary = 0`, recomputes modular ranks, and checks cycles modulo
boundaries. Integral homology is deferred because the current Smith-normal-form
result does not expose the left and right transformations required to bind
generators to the original chain bases.

Persistent homology requires a separate GUDHI provider spike that binds exact
input filtration values through simplex identities instead of trusting
floating backend coordinates. Low-dimensional manifold operations remain
Regina-specific later candidates.

## Rejected and deferred scope

The accepted records do not authorize:

- generic `probability.solve`, `discrete.solve`, `geometry.solve`, or
  `topology.solve` capabilities;
- mechanical wrappers for backend functions;
- floating Delaunay results presented as exact;
- producer replay presented as independent verification;
- truncated enumeration presented as complete;
- finite Gaussian checks presented as an all-parameter theorem;
- diagonal-only Smith form presented as certified integral homology; or
- GPL or C++ providers silently added to the core runtime.

## Implementation handoff

The four candidate objects in the status overlay contain:

- the atomic outcome and provisional IDs;
- recurrence or fundamental-primitive evidence;
- nearby portfolio entries and the non-duplicative delta;
- provider and independent-verification boundaries;
- public reproduction assets;
- a falsifiable evaluation hypothesis;
- remaining contract and checker obligations; and
- a pointer to the exact catalog and policy snapshot.

Implementation must return a new `stage=implementation,status=complete`
handoff. Checker and evaluation work create their own stage records instead of
silently strengthening the discovery evidence.
