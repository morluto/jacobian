# Fixed-registry graph invariant batches

[Documentation home](../../../index.md) · [Capability surface](../../tools.md)

`graph.compute.properties@2` is the canonical batch operation corresponding to
the inventory concept `graph.invariants.compute`. The inventory name is not an
installed alias.

## Supported registry

Registry version `1` contains exactly these invariant names:

```text
average_eccentricity
bipartite
connected
degree_sequence
diameter
eccentricities
girth
harmonic_index
havel_hakimi_trace
independence_number
maximum_degree
minimum_degree
order
radius
residue
size
tree
triangle_count
triangle_frequencies
```

The complete registry is exposed as `x-supported-invariants` in the input
schema and as `supported_invariants` in every completed output and batch
artifact. An invariant name that satisfies the input syntax but is absent from
this registry is retained with `UNSUPPORTED` status. It is not silently
ignored and does not abort other requested computations.

## Request and per-invariant results

The request supplies one compatible simple-undirected-graph artifact URI and
between one and 32 unique invariant names. Results are sorted by invariant
name. Every requested name receives exactly one result:

| Status | Meaning | Value | Backend |
| --- | --- | --- | --- |
| `COMPUTED` | The registered exact operation completed. | Present; JSON `null` is valid for invariants such as acyclic girth. | Exact producing backend is named. |
| `NOT_APPLICABLE` | The invariant is registered but undefined for this graph, such as diameter on a disconnected graph. | `null` | Attempted backend and a bounded explanation are retained. |
| `UNSUPPORTED` | Registry version `1` does not contain the requested name. | `null` | No backend is claimed. |

`properties` remains in the output and batch artifact as a compatibility
projection containing only computed values in the version-1 shape. New
consumers should use `results`, which preserves all terminal outcomes
uniformly.

## Artifacts and assurance

Each per-invariant result is stored in its own
`jacobian.graph-invariant-result@1` artifact with the source graph as parent.
The `jacobian.graph-property-batch@2` artifact binds the source graph, registry
version, supported and requested names, backend version, per-invariant results,
and their artifact URIs. The capability relationship targets both the batch
artifact and every individual result artifact.

A completed batch is `COMPLETE` when every requested name has one of the three
terminal statuses above. Completeness does not mean every requested invariant
was mathematically applicable or supported. Computation remains `COMPUTED`;
neither NetworkX nor the producing adapter independently verifies its own
results.

## Relationship to `graph.invariant.*`

Standalone `graph.invariant.*` capabilities are not aliases of the batch.
They accept inline bounded graphs, may expose operation-specific witnesses or
budgets, and in some cases use a different backend:

| Standalone invariant | Batch registry relationship |
| --- | --- |
| `graph.invariant.independence_number.compute` | Same mathematical value; different input and artifact contract. |
| `graph.invariant.girth.compute` | Same invariant; the standalone operation uses `0` for acyclic graphs while the batch uses `null`. |
| `graph.invariant.diameter.compute` | Same invariant; the standalone operation uses `-1` for disconnected graphs while the batch reports `NOT_APPLICABLE`. |
| `graph.invariant.clique_number.compute` | Not in registry version `1`. |
| `graph.invariant.chromatic_number.compute` | Not in registry version `1`; it has a bounded Z3 search and explicit `UNKNOWN` outcome. |
| `graph.invariant.edge_connectivity.compute` | Not in registry version `1`. |
| `graph.invariant.vertex_connectivity.compute` | Not in registry version `1`. |
| `graph.invariant.is_eulerian.compute` | Not in registry version `1`. |
| `graph.invariant.spanning_tree_count.compute` | Not in registry version `1`. |
| `graph.invariant.maximum_matching.compute` | Not in registry version `1`. |

Adding those operations to a later registry requires an explicit contract
decision about semantics, bounds, and result normalization. Version `2` does
not add or duplicate their algorithms.
