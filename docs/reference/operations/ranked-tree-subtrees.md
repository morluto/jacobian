# Ranked-tree subtrees

`ranked_tree.subtree.compute` selects a node by its zero-based child-index
position and returns the rooted subtree at that node. The empty position `[]`
selects the root, while an address that chooses a missing child is a domain
error.

The result retains the complete source tree and the selected position alongside
the subtree. `RankedTree` stores integer symbol identities and child structure;
it does not itself carry a separate ranked alphabet. Keeping the source in the
result therefore preserves the exact ambient tree identity and makes the
subtree's provenance explicit.

Before returning a value, the operation validates and bounds the whole source
tree (at most 4,096 nodes and depth 128), charges traversal and path work, and
admits the canonical output envelope. The output bound accounts for retaining
both the source tree and its selected subtree. The operation returns no partial
subtree on rejection.
