# Small exact graph reliability

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`probability.graph_reliability.connection_probability.compute` computes the
exact probability that two stated terminals are connected in a bounded
undirected graph with independent rational edge-open probabilities. Its result
includes the finite subset calculation needed to inspect that probability.

The complete graph and probabilities are request values, and the probability is
a response value. The operation has no stored graph source or verification
record.
