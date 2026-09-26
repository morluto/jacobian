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

The request admits the combined canonical-validation and profile work, degree
cells, and result bytes from the declared order and class count before nested
card parsing begins. Native calls perform the same combined admission before
defensively validating the untrusted card value, including its exact
permutation canonical form. The catalog parser then establishes that canonical
form once, and its adapter runs the profile kernel without repeating resource
admission or canonicalization.

`graph.deck.isomorphism_classes.compute` consumes a complete
`VertexDeletionFamily` and returns canonical representatives, exact class
multiplicities, a class index for every source card, and a vertex permutation
from each card's ordered vertex axis to its representative's `v00`, `v01`, ...
axis. The profile retains the source-bound family so every returned map can be
checked against the original card edges. It admits at most 10 source vertices,
including source-family validation, permutation canonicalization, map cells,
and output bytes. Empty and one-vertex sources retain zero-order cards and empty
permutations. Classifying the finite supplied deck does not reconstruct the
source graph or assert any reconstruction conjecture.

`graph.deck.edge.isomorphism_classes.compute` provides the corresponding exact
profile for a complete `EdgeDeletionFamily`. It returns canonical class
representatives, multiplicities, card indices, deleted source edges, and a
permutation from each card's retained source-vertex axis to its representative's
`v00`, `v01`, ... axis. The source-bound family keeps each mapping tied to the
edge deletion that produced the card. Admission charges exact permutation
canonicalization, source-family validation, map validation, and serialized
output; the profile supports at most 10 source vertices and a shared 2,000,000
work-unit bound. An empty edge deck on an edgeless graph has no classes or maps.
This operation classifies supplied cards and makes no graph reconstruction
claim.
