# Link diagram disjoint union

`link_diagram.disjoint_union.compute` forms a tagged disjoint union of oriented
link diagrams.

The result is source-transporting: it returns the input diagrams, the union
diagram, and explicit maps for crossings, darts, arcs, and free loops, so a
caller can locate any feature of the union back in the diagram it came from.
Crossing and free-loop counts are summed across the inputs and admitted
against the link crossing bound before the union is constructed.

## Scope

This operation builds the union only. It does not compute invariants of the
result, and it does not assert that the inputs are related. For the invariants
of a single diagram — the Jones and Alexander polynomials, the determinant, the
Goeritz matrix, and the Seifert circles — see the link diagram pages.

`link_diagram.signature.compute` and `link_diagram.blackboard_graph.compute` are
described on their own page and are not covered here.
