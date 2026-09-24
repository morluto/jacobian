# Cubical boundary subcomplexes

[Topology operations](index.md) · [Tool surface](../../tools.md)

`topology.cubical_complex.boundary_subcomplex.compute` returns the boundary
subcomplex of a finite pure cubical complex in the integer lattice. The input
must have positive dimension, one ambient coordinate axis, and every cell must
be a face of a top-dimensional cell.

A codimension-one face is exposed when it is incident to exactly one
top-dimensional cell. The operation returns those exposed facets and their
complete downward closure. A facet shared by two top-dimensional cells is
interior and is omitted. The convention is combinatorial and exact; it does
not infer a manifold boundary or resolve nonmanifold singularities.

The result retains the face-closed source complex and represents its boundary
using the same `CubicalComplex` value. This value retains the ambient dimension
even when the boundary has no cells, as for the cubical surface formed by the
six faces of a 3-cube. Requests remain nonempty; only the derived subcomplex may
be empty.

Before closure, the operation bounds coordinate digits, the number of face
candidates, top-cell incidence and canonicalization work to 8,000,000 work
units, and the encoded source-plus-boundary result to 8 MiB. Current limits
are 5,000 input generators, ambient dimension 10, and 64 decimal digits per
lattice coordinate.

This is the standard combinatorial boundary convention for a finite abstract
cubical complex: its boundary facets are codimension-one faces contained in
exactly one maximal face ([Joswig and Schröder, *Neighborly Cubical Polytopes
and Spheres*, §2.2](https://www.mathematik.tu-darmstadt.de/media/mathematik/forschung/preprint/preprints/2424.pdf)).
Manifold or pseudomanifold hypotheses are not inferred; the result reports the
exposed-facet subcomplex even when that subcomplex has singularities.
