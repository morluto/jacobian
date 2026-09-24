# Classical link diagram operations

[Topology operations](index.md) · [Tool surface](../../../tools.md)

These operations consume an `OrientedLinkDiagram`, a bounded classical diagram
with a cyclic order of four darts at each crossing, explicit over/under strands,
directed arcs, and checked crossing signs. The diagram is a presentation; these
operations do not decide whether two presentations are isotopic.

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
matrix and determinant are diagram-level values; this operation does not
compute a signature correction or make a link-equivalence decision.

For the positive Hopf link, the checkerboard graph has two vertices joined by
two positive parallel edges. Its Laplacian is
`[[2, -2], [-2, 2]]`, and deleting one vertex gives the `1 × 1` Goeritz matrix
`[[2]]`.
