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
This makes relabelings serialize to the same value. Deserialization enforces
the structural representation (fixed labels, valid ordered edges, and strictly
ordered distinct edge-tuples), but deliberately does not recompute permutation
minimality: doing so would repeat factorial mathematical work outside the
operation's admission and deadline. Canonicality is guaranteed by the
constructing operation, not by decoding an untrusted value.

This anonymous value stores neither a source graph nor deletion identifiers.
It does not assert that the card multiset is realizable as a vertex or edge
deck. Source enumeration and reconstruction are separate mathematical
questions and are not performed by this operation.
