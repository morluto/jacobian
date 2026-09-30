# Link diagram disjoint union

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`link_diagram.disjoint_union.compute` forms a tagged disjoint union of oriented
link diagrams.

The result is source-transporting: it returns the input diagrams, the union
diagram, and explicit maps for crossings, darts, arcs, and free loops, so a
caller can locate any feature of the union back in the diagram it came from.
Crossing and free-loop counts are summed across the inputs and admitted
against the link crossing bound before the union is constructed.

## Scope

This operation builds the union only. It does not compute invariants of the
result, and it does not assert that the inputs are related. It does not compute
the Jones or Alexander polynomial, the determinant, the Goeritz matrix, or the
Seifert circles of the union either.

`link_diagram.signature.compute` and `link_diagram.blackboard_graph.compute` are
published operations on a single diagram, but they have no operation reference
page of their own. Discover them with `math.find`, or read the tool surface
reference.
