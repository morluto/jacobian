# Induced filtration on homology

[Topology operations](index.md) · [Tool surface](../../tools.md)

`homological.filtered_chain_complex.homology_filtration.compute` returns
exact homology bases over the admitted bounded prime fields and the image of
each source filtration level in every homology group.

Each degree retains a basis of cycles, incoming boundaries, and homology
representatives in the original chain basis. Each filtration image is given in
coordinates on that retained homology basis. For every image basis vector, the
result also carries a cycle in the corresponding filtration level and an
incoming chain whose boundary is the difference between that cycle and its
homology-basis representative. These data let a caller replay both the source
cycle and the quotient-class equality.

The returned objects are nested image subspaces. They do not choose a direct
sum splitting of homology from associated-graded pieces. The initial public
envelope admits `GF(p)` with `p <= 2^31 - 1`; rational-coefficient output
growth remains a separate admission problem.
