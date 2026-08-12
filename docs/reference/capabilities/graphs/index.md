# Graph capability references

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

- [Graph distance matrix](graph-distance-matrix.md)
- [Fixed-registry graph invariant batches](graph-invariant-batch.md)
- [Maximum-matching certificate and verification](graph-maximum-matching.md)
- [Graph diameter and radius verification](graph-metric-verification.md)
- [Exact weighted minimum spanning tree](graph-minimum-spanning-tree.md)
- [Small exact graph reliability](graph-reliability.md)
- [Declared graph-symmetry orbits](graph-symmetry-orbits.md)

`graph.isomorphism.verify` version 2 returns `first_violation` with every
independently verified `FALSE` mapping verdict. It deterministically reports
the first source-domain mismatch, target-bijection mismatch, or unordered
vertex pair whose source and mapped-target adjacency differ. This typed
counter-witness is absent for `TRUE` and `UNKNOWN` conclusions.
