# Finite-poset order complex

[Topology operations](index.md) · [Tool surface](../../tools.md)

`topology.poset.order_complex.compute` takes an existing canonical
`FinitePoset` and returns the finite abstract simplicial complex whose vertices
are the poset elements and whose faces are all nonempty chains. Thus every
maximal chain is a facet, and its subsets provide every shorter chain. This is
the standard order-complex definition; for a simplicial complex `K`, the order
complex of its nonempty face poset is its barycentric subdivision
([Matoušek, *Using the Borsuk–Ulam Theorem*, §1.7](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/matousek1x.pdf)).

The operation rechecks the source poset's retained order claims at its admitted
consumer boundary. It returns the same `FinitePoset`, a canonical
`FiniteSimplicialComplex`, an identity `vertex_elements` axis, and the maximal
chains in poset order. The complex reuses the poset labels as vertex labels;
these are not renumbered or merged.

Before enumerating chains, the operation counts all nonempty chains and their
total vertex incidences by greatest element. It also counts maximal chains in
the cover graph. These exact counts preflight the finite-complex face, facet,
dimension, work, and retained-cell bounds. A poset with a long chain or too many
chains is rejected before chain materialization; no prefix is returned.

An antichain has one singleton facet per element, and a one-element poset has
one vertex. An empty relation is therefore supported. The empty poset is
outside the current `FiniteSimplicialComplex` carrier, which requires at least
one vertex and one facet, so it is rejected explicitly.

Note that the maximal chains of a face poset are maximal *flags*, not faces: a
subdivided triangle has six maximal chains, one per small triangle. Build the
face poset yourself from a complex's faces when you need the barycentric
subdivision, and compare it against
`topology.simplicial.barycentric_subdivision.compute`.
