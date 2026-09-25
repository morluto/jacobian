# Graph deck values

The graph-deck operations distinguish source-bound deletion families from
anonymous multisets of graph cards. `graph.deck.from_cards.construct` accepts a
declared `card_order` and an unordered tuple of simple graphs of that order.
Repeated isomorphic cards contribute to an exact positive multiplicity. The
empty multiset keeps its declared order because it has no cards from which to
infer one.

The result is `AnonymousGraphCardMultiset`: an ordered tuple of distinct
isomorphism classes with fixed-width labels `v00`, `v01`, and so on. Its
canonical form is the lexicographically least zero/one adjacency vector over
all vertex permutations, using pair order `(0,1), (0,2), ..., (n-2,n-1)`.
This makes relabelings serialize to the same value. Deserialization checks the
same permutation-minimality condition and rejects repeated isomorphism
classes. That check admits work before canonicalization, charging every
permutation for adjacency-vector generation and a worst-case full-vector
comparison (which is reached when candidates tie, such as for empty or complete
graphs), along with per-order setup. Operation construction uses a trusted
kernel path after its own request admission to avoid repeating the computation.

This anonymous value stores neither a source graph nor deletion identifiers.
It does not assert that the card multiset is realizable as a vertex or edge
deck. Source enumeration and reconstruction are separate mathematical
questions and are not performed by this operation.

## Anonymous vertex-deck edge count

`graph.deck.anonymous_vertex.edge_count.compute` consumes an
`AnonymousGraphCardMultiset` directly, so its input composes with
`graph.deck.from_cards.construct` after a caller has supplied the cards. For a
vertex deck of an n-vertex simple graph with n >= 3, each source edge survives
exactly n-2 vertex deletions. Summing the card edge counts therefore gives
`(n-2)|E|`, and division recovers the exact source edge count. This is the
edge-count instance of Kelly's subgraph-counting identity; the operation uses
the direct incidence argument and does not claim that divisibility alone
proves deck realizability. See [Kelly's 1957 paper](https://doi.org/10.2140/pjm.1957.7.961)
and the [Bondy–Hemminger reconstruction survey](https://doi.org/10.1002/jgt.3190010306).

The operation admits card order at most seven so that exact canonical-form
revalidation of all distinct cards stays within the permutation-work bound.
Order zero is represented by the empty multiset of zero-vertex cards, and
order one by one empty card; both have edge count zero and no divisor. Order
two is rejected: its two one-vertex cards are identical whether the source has
zero edges or one. For orders at least three, a nondivisible total is rejected
as inconsistent with a vertex deck; a divisible total returns the unique
candidate edge count conditional on the supplied multiset being a realizable
vertex deck.
