# Ranked-tree positions

`ranked_tree.positions.compute` returns one zero-based child-index path for
every node of a finite ranked tree. The root has position `[]`; child `i` of a
node at `p` has position `p + [i]`. Rows are ordered in root-first preorder.

The result retains the source `RankedTree` and the complete position tuple.
The operation does not need an automaton or ranked signature: `RankedTree`
already bounds each node's symbol and number of children, while the position
operation independently admits at most 4,096 nodes and depth 128. It prices
the total path-coordinate count and canonical result bytes before constructing
the output list. A resource rejection does not return a partial list.

These positions are structural addresses, not symbol labels or automaton
states. They can be reused by later tree operations such as subtree extraction
and replacement.
