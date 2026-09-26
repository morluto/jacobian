# Exact graph deletion decks

`graph.deck.vertex_deleted.compute` and `graph.deck.edge_deleted.compute`
construct complete source-bound deletion families. A vertex card omits exactly
its bound vertex. An edge card retains the entire source vertex domain,
including isolated vertices.

`graph.deck.unlabelled.compute` groups edge cards by exact graph isomorphism.
`graph.deck.vertex.unlabelled.compute` does the same for vertex cards. Both
return one representative card, its positive multiplicity, and the complete
source-card index set for each class; the edge quotient's deleted source edges
are recovered from the card indices on the source edge axis. Isomorphism
classes are determined by canonicalizing each card to the least adjacency bit
word over all vertex permutations, a complete invariant rather than a degree
sequence or hash approximation, so the aggregate exact permutation work is
preflighted before any canonical form is computed. The edge quotient admits
per-card permutation canonicalization over the full source order. The
source-bound vertex quotient admits source orders through eight under
`n*(n-1)!*(1+(n-1)+binom(n-1, 2)) <= 2,000,000` canonicalization work.

`graph.deck.vertex.anonymous.compute` forgets labels after authenticating the
complete source-bound family. It admits source orders through seven because it
also reserves the work for a self-comparison:
`2*n*(n-1)!*((n-1)+2*binom(n-1, 2)) <= 2,000,000`.

`graph.deck.vertex.induced_subgraph_count.compute` reconstructs the number of
induced copies of a caller-supplied pattern `H` when `|V(H)| < n`, where `n` is
the source order in the deck. It reuses the exact induced vertex-subset count
convention, so each copy means one vertex subset whose induced graph is
isomorphic to `H`, independent of `H`'s automorphisms. Every such copy survives
in exactly `n - |V(H)|` cards. The operation sums card contributions weighted
by deck multiplicity and divides by that exact factor. It admits deck
validation, canonicalization, card-count work, and the result's aggregate
vertex-label echo allocation before reconstructing the quotient or building
contributions. Its output identifies the pattern, deck, each isomorphism-class
contribution, weighted total, divisor, and quotient.

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
contribution breakdown as the induced operation. Work and the result's label
echo allocation are admitted
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

## Anonymous card multiset equality

`graph.deck.vertex.anonymous.compute` is the typed bridge from
`VertexDeletionFamily` to `AnonymousGraphCardMultiset`. It authenticates the
complete source-bound family, forgets source vertex labels and deletion keys,
then returns fixed-axis canonical isomorphism-class representatives with exact
card multiplicities. The result composes unchanged with anonymous-deck
operations, including equality; it retains no source graph and makes no deck
realizability or reconstruction claim. Canonicalization work and result size
are admitted before family replay or permutation search.

`graph.deck.anonymous.equal.decide` compares two
`AnonymousGraphCardMultiset` values. It returns true exactly when both values
have the same card order and the same multiplicity for every graph-isomorphism
class. Card labels and class row order do not identify cards; the classes
represent the multiset after each card is independently relabelled.

The operation verifies each bounded representative's fixed axis and edge
shape, then computes an exact permutation canonical form for every class on
both sides. This consumer check is needed because structural decoding alone
does not establish that a supplied representative is the canonical member of
its isomorphism class. It admits aggregate work across both inputs before any
permutation search, with a 2,000,000-unit limit. The result concerns only
multiset equality; it makes no claim that either multiset is realizable as a
graph deck, and equal decks do not imply source-graph isomorphism.
