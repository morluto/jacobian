# Classical link diagram operations

[Topology operations](index.md) · [Tool surface](../../../tools.md)

These operations consume an `OrientedLinkDiagram`, a bounded classical diagram
with a cyclic order of four darts at each crossing, explicit over/under strands,
directed arcs, and checked crossing signs. The diagram is a presentation; these
operations do not decide whether two presentations are isotopic.

## Disjoint union

`link_diagram.disjoint_union.compute` combines one to 64 source diagrams into
one diagram. It assigns deterministic labels on source-indexed crossing and
dart axes, preserves crossing signs and directed arc pairings, and returns
complete source-to-target maps for crossings, darts, arcs, and crossing-free
components. The source index and each source label together identify a crossing
or dart even when input diagrams reuse the same labels. Crossing-free loops
have no labels in `OrientedLinkDiagram`, so the result maps each loop by its
source index and local loop index.

Admission bounds the union to at most 64 crossings, 64 free loops, and an
estimated 8 MiB serialized result before output construction. Work is linear in
the admitted source diagram and transport axes. A one-input union is allowed
and produces the same canonical tagged relabeling.

## Signed checkerboard graph

`link_diagram.blackboard_graph.compute` returns the source-bound signed Tait
graph for one deterministic checkerboard coloring. Region cycles are obtained
from the dart rotation and arc involution. The region containing the least
boundary dart is shaded, and colors alternate across projection arcs. Each
shaded region is a graph vertex, and each crossing
contributes one signed edge between its two opposite-color corner regions.
Crossing IDs remain edge IDs, so parallel edges are preserved; equal endpoint
regions are permitted and encode loops. The result retains every planar region
boundary and every crossing-edge correspondence with corner indices. Each edge
stores its Tait sign: `+1` when its shaded corners are the overpassing pair and
`-1` otherwise.

This matches the classical Tait construction: a vertex represents each region
of one checkerboard color, each crossing gives an edge between the adjacent
regions, and the edge retains the Tait sign. See [Moffatt, “Partial duals
of plane graphs, separability and the graphs of knots,” §3.1](https://msp.org/agt/2012/12-2/agt-v12-n2-p16-s.pdf).

The operation admits connected, nonempty crossing projections with at most 64
crossings. Its work is linear in the dart and crossing axes; its conservative
512 KiB result estimate (including JSON escaping for the label bound) is checked
before face traversal. Crossing-free diagrams and disconnected crossing
projections are rejected because this representation currently selects one
connected plane graph.

## Goeritz matrix

`link_diagram.goeritz_matrix.compute` consumes the same typed checkerboard graph
and forms its signed Laplacian. It deletes the final shaded-region row and
column, returning the exact integral reduced matrix and its absolute
determinant. The matrix operation has a tighter 32-crossing envelope. The
matrix and determinant are diagram-level values and do not by themselves give
the link signature.

For the positive Hopf link, the checkerboard graph has two vertices joined by
two positive parallel edges. Its Laplacian is
`[[2, -2], [-2, 2]]`, and deleting one vertex gives the `1 × 1` Goeritz matrix
`[[2]]`.

## Oriented link signature

`link_diagram.signature.compute` composes the reduced Goeritz matrix with exact
rational inertia and the Gordon–Litherland correction term. Under this
module's Tait convention, each crossing incidence is `+1` when the shaded
corners are the overpassing pair and `-1` otherwise. The oriented crossing
sign is already checked against the diagram's directed arcs. A crossing is
type II exactly when the product of that crossing sign and its Tait incidence
is `-1`; the correction is the sum of incidences over type-II crossings. The
returned signature is

```text
number of positive Goeritz directions
- number of negative Goeritz directions
- type-II correction.
```

The operation retains the source diagram, Goeritz matrix when crossings are
present, exact inertia result, and each crossing's type and correction
contribution. Crossing-free unlinks return zero using the empty-matrix inertia
convention. Nonempty projections must be connected and have at most 32
crossings, matching the Goeritz matrix bound. It does not test link equivalence.
The defining relation is the Gordon–Litherland formula for an oriented link;
see [Gordon and Litherland, “On the signature of a link”](https://doi.org/10.1007/BF01609479)
and the explicit checkerboard sign conventions in Cimasoni and Ferretti,
[§2.2](https://www.unige.ch/~cimasoni/Kashaev-signature.pdf).
