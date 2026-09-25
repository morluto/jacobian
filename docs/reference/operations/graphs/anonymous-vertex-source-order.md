# Anonymous vertex-deck source order

`graph.deck.anonymous_vertex.source_order.compute` recovers the source order
from the complete anonymous multiset of vertex-deleted graph cards. For a
nonempty graph of order `n`, every card has order `n - 1` and the deck has `n`
cards, with multiplicity. The order-zero graph has no cards; the order-one
graph has one empty card. These conventions distinguish both small cases.

The operation rejects an incomplete multiset when its declared card order and
total multiplicity do not fit one of those cases. It retains the exact
anonymous card multiset with the returned source order, so the value composes
with edge-count and degree-multiset reconstruction. Satisfying these necessary
conditions does not establish that the cards are realizable as the deck of a
graph.

The card order is bounded by ten, and source order by eleven. Work is linear
in the number of isomorphism classes, after the typed multiset's bounded
canonical representation has been admitted.

The contract follows the standard definition of a graph deck as the family of
all unlabelled one-vertex-deleted subgraphs, one per source vertex. See
[*A Graph Reconstructor's Manual*](https://www.cambridge.org/core/books/abs/surveys-in-combinatorics-1991/graph-reconstructors-manual/C120B6148CB00DEC77BD7DA51247BF49).
