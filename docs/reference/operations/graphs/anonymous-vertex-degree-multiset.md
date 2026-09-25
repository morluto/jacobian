# Anonymous vertex-deck degree multiset

`graph.deck.anonymous_vertex.degree_multiset.compute` consumes the existing
`AnonymousGraphCardMultiset` carrier. It composes with the anonymous edge-count
operation and returns the sorted source degree multiset together with that
typed edge-count result.

If the supplied cards are the complete vertex-deleted deck of a simple graph
`G`, then for each deleted vertex `v`,

```text
deg_G(v) = |E(G)| - |E(G-v)|.
```

The operation obtains `|E(G)|` from the card-edge incidence identity, derives
one degree for each card with its multiplicity, and checks the degree bounds,
handshake identity, and graphicality. These conditions are necessary; they do
not certify that an arbitrary anonymous card multiset is realizable as a
vertex deck. The degree sequence is a reconstructible graph parameter; see
[Myrvold, “The degree sequence is reconstructible from n − 1 cards” (1992)](https://doi.org/10.1016/0012-365X(92)90053-I).

The underlying card-edge relation is exact because deleting any one vertex
removes precisely its incident edges. The edge-count prerequisite uses the
standard `(n-2)|E(G)|` double count. Output order is nonincreasing; the order
zero graph has an empty degree tuple. Order two remains unsupported because
its anonymous vertex deck does not determine the source edge count, and hence
does not determine the source degree multiset.

The result retains the edge-count value and therefore the original card
multiset, card order, and source order. A consumer that accepts a separately
authored or modified result as a claim should recheck that its degree entries
are obtained by subtracting each card's edge count from the retained source
edge count.
