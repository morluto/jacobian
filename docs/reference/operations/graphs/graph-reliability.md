# Small exact graph reliability

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`probability.graph_reliability.connection_probability.compute` computes the
exact probability that two stated terminals are connected in a bounded
undirected graph with independent rational edge-open probabilities. Its result
includes the finite subset calculation needed to inspect that probability.

`probability.graph_reliability.all_terminal.compute` instead computes the
probability that the spanning subgraph on every declared vertex is connected.
Every edge has the same exact rational open probability. The source-bound
result includes the connected-spanning-subgraph counts by open-edge cardinality,
so its probability reconstructs as
`sum(c[k] * p^k * (1-p)^(m-k))`. The operation accepts nonempty canonical
simple graphs with at most 20 edges, bounding the complete enumeration by
`2^20` states. Because the exact result retains its source, admission accounts
for the graph and the
complete coefficient-profile cardinality directly.

These are different events: two chosen vertices can remain connected while an
isolated third vertex makes all-terminal reliability fail.

Both operations are stateless. The all-terminal result retains its graph and
uniform edge probability and coefficient profile. The kernel performs the
enumeration once; parsing the result checks bounded shape, not the defining
enumeration. Independent mathematical evidence belongs in the owning tests.
