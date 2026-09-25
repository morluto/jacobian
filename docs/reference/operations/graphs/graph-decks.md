# Exact graph deletion decks

`graph.deck.vertex_deleted.compute` and `graph.deck.edge_deleted.compute`
construct complete source-bound deletion families. A vertex card omits exactly
its bound vertex. An edge card retains the entire source vertex domain,
including isolated vertices.

`graph.deck.unlabelled.compute` groups edge cards by exact graph isomorphism.
`graph.deck.vertex.unlabelled.compute` does the same for vertex cards. Both
return one representative card, its positive multiplicity, and the complete
source-card index set for each class. Isomorphism classes are determined by
exact pairwise graph isomorphism, not degree sequences or hashes. Quotienting is
limited to at most 10 source vertices and a preflighted work bound. The vertex
quotient currently admits orders through eight; order nine and above exceed its
exact permutation-canonicalization budget. The edge quotient has its own
comparison bound.

`graph.deck.from_cards.construct` accepts an unordered finite multiset of cards
of one declared order and stores each exact graph-isomorphism class once with
its positive multiplicity. Its canonical representatives make card relabeling
irrelevant while preserving repeated-card counts. The empty multiset retains
its declared order. Deserialization checks the fixed axis, graph-edge structure,
row ordering, and multiplicity bounds; it does not prove that each supplied row
is a permutation-minimal representative or that different rows are
nonisomorphic. The degree invariant profile uses only graph degrees, so it does
not require permutation canonicalization.

`graph.deck.vertex.induced_subgraph_count.compute` reconstructs the number of
induced copies of a caller-supplied pattern `H` when `|V(H)| < n`, where `n` is
the source order in the deck. It reuses the exact induced vertex-subset count
convention, so each copy means one vertex subset whose induced graph is
isomorphic to `H`, independent of `H`'s automorphisms. Every such copy survives
in exactly `n - |V(H)|` cards. The operation sums card contributions weighted
by deck multiplicity and divides by that exact factor. Its output identifies
the pattern, deck, each isomorphism-class contribution, weighted total,
divisor, and quotient.

This induced operation has its own ID because the existing exact graph count
owner counts induced vertex subsets. It does not claim the ordinary
non-induced subgraph-copy convention of Kelly's original lemma.

`graph.deck.vertex.subgraph_count.compute` counts ordinary copies of a proper
pattern `H`: a copy is a pair consisting of a vertex subset and an edge subset
on it whose graph is isomorphic to `H`. Additional source edges on the chosen
vertices are allowed. Thus copies differ from induced vertex subsets and from
injective embeddings (which count each copy once for every automorphism of
`H`). The operation counts embeddings on each card and divides by the pattern's
automorphism count, weights by exact deck-class multiplicity, then divides by
`n - |V(H)|` using Kelly's identity
`sum_v copies(H, G-v) = (n - |V(H)|) copies(H, G)`: each fixed copy survives
the cards deleting any vertex outside its chosen vertex set. It returns the same source-bound
contribution breakdown as the induced operation. Work and output are admitted
before deck validation, canonicalization, or assignment enumeration; source
order is limited to the vertex quotient's exact envelope.

The ordinary-copy meaning and double-counting identity follow the standard
Kelly lemma: see Groenland, Guggiari, and Scott, [*Size reconstructibility of graphs*]
(https://onlinelibrary.wiley.com/doi/full/10.1002/jgt.22616), Section 3, which
states that every proper subgraph count is reconstructible from the full deck.

The vertex quotient consumes a complete `VertexDeletionFamily`; at the consumer
boundary it reestablishes that family from its source before making
isomorphism comparisons. It therefore rejects dropped, duplicated, or altered
cards even if a caller bypassed model validation. The result retains the input
family so its source and every class index remain reconstructible after
serialization.

These operations construct and summarize decks. They do not decide whether a
graph can be reconstructed from its deck.
