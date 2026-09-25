# Finite simplicial topology

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The topology family owns bounded finite simplicial-complex values and their
exact direct operations:

- `topology.simplicial_complex.canonicalize`
- `topology.simplicial_complex.chain_complex.compute`
- `topology.simplicial_complex.induced_subcomplex.compute`
- `topology.simplicial_complex.f_vector.compute`
- `topology.simplicial_complex.face_enumerator.compute`
- `topology.simplicial_complex.g_vector.compute`
- `topology.simplicial_complex.clique_from_graph.compute`
- `topology.poset.order_complex.compute`
- `topology.simplicial_complex.one_skeleton.compute`
- `topology.simplicial_homology.compute`
- `topology.simplicial_homology.integral.compute`

The integral operation wraps the same chain-complex-owned `HomologyResult`
returned by `chain_complex.homology.compute`. It retains the canonical `ZZ`
chain value, Smith transformations, source-basis cycles, torsion invariant
factors, and bounding chains. Reduced chains represent the augmentation as the
ordinary differential from degree 0 to a rank-one group in degree -1. The
operation does not create a durable complex, certificate record, or checker
session.

The `f_vector` operation uses the convention `(f_-1, f_0, ..., f_d)`, with
`f_-1 = 1` for the empty face. For `{∅}`, its nonempty-face `f_vector` is
`()`, its dimension is `-1`, and its empty face stays implicit. The operation
also returns the corresponding `h`-vector and Euler characteristic. Reduced
simplicial homology includes `H̃_-1({∅}) = Z` and zero degree-zero homology;
unreduced degree-zero homology is zero. The `g_vector` operation returns
`g_0 = 1` and
`g_i = h_i - h_(i-1)` through `i = floor((d+1)/2)`, along with the exact
f- and h-vectors it uses. These are finite transforms of face counts; neither
operation asserts that the complex is a sphere or manifold, or that g-vector
entries are nonnegative.

The `face_enumerator` returns the canonical `IntegerPolynomial`
`F_K(t) = sum_{sigma in K} t^|sigma|`, including the empty face as constant
term one. Its descending-degree coefficients are the reversed nonempty
f-vector followed by that constant term, so the result composes directly with
integer-polynomial operations. Candidate subface generation is admitted before
expansion; a unique-face counter stops insertion at the finite-complex face
ceiling and the admitted closure is reused to construct the polynomial.

`clique_from_graph.compute` accepts an indexed simple graph and returns its
flag complex as the existing `CliqueResult`, including isolated vertices as
singleton facets. It uses deterministic labels `v0`, `v1`, and so on. The
operation admits 1 through 8 vertices, since the carrier represents dimensions
through seven. At the largest admitted size, a complete graph returns the
7-simplex with 255 nonempty faces. The null graph is outside that carrier
because it has no nonempty simplices.

The barycentric-subdivision result binds every output vertex to its source face
and every maximal output simplex to its increasing strict chain of source
faces. Admission counts maximal chains before enumerating them (at most 128)
and checks the complete output face closure (at most 2048 nonempty faces).
This allows low-output cases such as a 32-point discrete complex while
rejecting a simplex whose order complex has too many maximal chains.

The reusable `topology.poset.order_complex.compute` operation applies the same
chain construction to any admitted `FinitePoset`, retaining the exact element
labels on the output vertex axis. See the [order-complex contract](../topology/order-complex.md)
for its exact face, dimension, maximal-chain, work, and output bounds.

`induced_subcomplex.compute` selects a subset (possibly empty) of the canonical
vertex axis and returns the full subcomplex whose faces are exactly the source
faces contained in that subset. Its face-image table maps retained faces to
the same vertex tuple and marks every removed source face with `null`. An empty
selection returns the canonical zero-vertex complex `{∅}`.

The `link` and `star` operations accept the implicit empty face as well as
nonempty faces. Its link and closed star are both the source complex. The
result carries the exact canonical target complex; in particular, the link of
a maximal face is the zero-vertex complex `{∅}`, represented by empty
nonempty-face axes and dimension `-1`. `link_is_empty` identifies that
canonical value; Jacobian's finite-complex carrier does not represent the
void complex.

## Minimal nonfaces

`topology.simplicial_complex.minimal_nonfaces.compute` returns all
inclusion-minimal vertex subsets not in the source complex. Each subset uses
the source's canonical vertex order; the result retains the complete
`FiniteSimplicialComplex` source, so its labels cannot be mistaken for subsets
of a different vertex domain. Rows are ordered first by cardinality, then
lexicographically. The empty set is the implicit face of every represented
nonvoid complex and is not returned.

The operation enumerates the powerset and tests immediate subfaces. Admission
bounds the candidate count at 16,384, the combined subset, closure, and facet
work at 430,000 candidate-and-label steps, the antichain
output using Sperner's bound, and the source-plus-result encoding at 2.5 MB.
Consequently, the current envelope admits at most 14 source vertices; 15
vertices are rejected before powerset expansion. Minimal
nonfaces are precisely the squarefree generators used to define the
Stanley–Reisner ideal, but this operation returns subsets and does not construct
a polynomial ring or ideal.

The canonical carrier stores nonempty faces only. The zero-vertex complex
`{∅}` uses empty vertex, facet, and face axes, `f_vector = ()`, and dimension
`-1`; its empty face remains implicit. The void complex, which has no faces
including the empty face, is not represented. On the represented domain, a
full simplex has an empty minimal-nonface antichain.

The cardinality bound follows from Sperner's theorem: an antichain of subsets
of an `n`-element set has at most `binom(n, floor(n/2))` members ([MIT OCW
combinatorics notes](https://ocw.mit.edu/courses/18-226-probabilistic-methods-in-combinatorics-fall-2022/mit18_226_f22_lec_full.pdf)).

`one_skeleton.compute` returns the exact graph on the canonical vertex axis as
an `IndexedSimpleUndirectedGraph`, which graph operations can consume unchanged.
The accompanying `vertex_labels` tuple maps graph indices to source labels,
and `edge_faces` is aligned with the graph edge list, preserving the exact
source 1-face for each edge. Isolated vertices remain in the graph axis. The
source complex's 64-vertex bound implies at most `binom(64, 2) = 2,016` graph
edges, below the indexed graph value's edge cap. Decoding checks that the
vertex axes agree and that the edge map matches the source's stored 1-face
axis; it does not replay the source complex's construction history.
