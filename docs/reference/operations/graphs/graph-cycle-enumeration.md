# Complete fixed-length cycle families

[Graph operations](index.md) · [Tool surface](../../tools.md)

The catalog exposes two exact operations for a canonical finite simple
undirected graph `G` and an integer `k >= 3`:

- `graph.cycle.fixed_length.enumerate` returns every simple `k`-cycle.
- `graph.cycle.chordless_fixed_length.enumerate` returns every induced
  (chordless) simple `k`-cycle.

Each cycle is represented once in lexicographically smallest dihedral form
(the smallest rotation in either orientation), and the family is sorted by
that tuple. The result retains `G`, `cycle_length`, `cycle_count`, and complete
source-axis indexes: each vertex and each source edge lists the zero-based
indices of incident cycles. A cycle uses distinct declared vertices and every
consecutive pair, including the closing pair, is a source edge.

The two operations are separate contracts: a simple cycle may contain a chord,
while a chordless cycle may not contain any nonconsecutive edge among its
vertices. The complete family must fit the operation's admitted traversal,
intermediate, retained-label, and result bounds. A graph with no such cycle,
or a requested length larger than the graph's cycle-bearing 2-core, returns an
empty family with all source vertex and edge axes retained. Lengths larger than
the graph order are therefore valid empty requests; malformed carriers and
requests outside the declared operation envelope are rejected before search.
