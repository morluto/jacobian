# Finite lattice loop-family holonomies

`lattice_gauge.loop_family.holonomies.compute` evaluates an explicitly
supplied finite tuple of based closed paths over one permutation-valued gauge
field. Each output entry retains its path, basepoint, and exact holonomy. The
result retains the source field once, so a family does not repeat the entire
lattice and link assignment per loop. Empty identity paths are allowed when
their basepoint is a source vertex; an empty family returns the same source
field with no loop entries.

For each path \(\gamma=e_1\cdots e_m\), the operation returns
\(\operatorname{Hol}_U(\gamma)=U_{e_1}\cdots U_{e_m}\) in traversal order,
using the exact inverse label for reverse steps. Every path must chain on the
source lattice and close at its basepoint. The operation evaluates only the
provided paths; it does not search for loops or choose a generating family.

Admission allows at most 128 loops, 256 steps per loop, and 4096 steps across
the family. The kernel admits the source field once and bounds the work by
\(V + E d^2 + 8E\ell + L + S(2d+\ell+1) + T\), where \(V\) is the vertex
count, \(E\) the edge count, \(d\) the permutation degree, \(L\) the number of
loops, \(S\) the aggregate path length, \(\ell=64\) the maximum label length,
and \(T\) the total length of source and path text labels. The label terms
cover bounded string checks and lookups. Before multiplying any labels, it also
bounds the
source-bound result by a conservative count of value cells and scalar text
units. These bounds are mathematical value bounds, not serialized-byte
limits.

The family result composes with the same source-bound path and permutation
holonomy values as `lattice_gauge.holonomy.compute`. It supports the bounded
permutation group \(S_d\), \(1\le d\le8\); finite multiplication-table and
rational SU(2) families have their own typed fields and are not coerced into
this representation.
