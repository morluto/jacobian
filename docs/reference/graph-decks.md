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

`graph.deck.card_invariant_profile.compute` consumes this carrier and returns a
histogram of nonincreasing degree multisets. It combines frequencies even when
different nonisomorphic card classes share the same degree multiset, and weights
each class by its exact input multiplicity. The result retains `card_order` and
the total card multiplicity, so an empty card multiset of order `n` produces an
empty histogram with order `n` and total zero. This invariant profile makes no
source-deck realizability claim.

The consumer admits canonical-validation work, degree-vector and edge-scan
work, histogram aggregation and ordering, materialized degree cells, and result
bytes before profile construction. Native calls
re-admit the untrusted card value, including its exact permutation canonical
form. The catalog adapter relies on the strict JSON parsing boundary to perform
that canonical validation once, then uses the admitted profile kernel without
repeating canonicalization.
