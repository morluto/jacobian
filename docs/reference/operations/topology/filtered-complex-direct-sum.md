# Direct sums of filtered chain complexes

`homological.filtered_chain_complex.direct_sum.compute` forms the finite
componentwise direct sum of two exact filtered chain complexes. Both inputs
must have the same coefficient field, chain-degree interval, and number of
filtration levels. At each degree and filtration level, the output subspace is
the direct sum of the two input subspaces; the output boundary is block
diagonal.

The result contains a `FilteredChainComplexRequest`, which can be passed
unchanged to the associated-graded and abutment operations. For spectral-page
computation, pass its `complex` and `filtration` fields and supply the required
`page` field. Filtered-homology composition is supported when the shared
coefficient field is a prime field; `QQ` results are not accepted there. It also returns the two coordinate inclusions. The
accepted coefficient fields are `QQ` and prime fields; integral filtered
complexes remain outside this operation's contract.

Admission bounds the output chain-group dimensions, differential cells,
filtration vector counts, total filtration scalar cells, and work for exact
source validation before constructing the direct sum. The output filtration
keeps the input spanning vectors, embedded into the corresponding summands;
it does not compute a reduced or canonical basis.

## Mathematical reference

Arapura defines filtered modules with direct sum filtration
`F^p(M ⊕ N) = F^pM ⊕ F^pN` and filtered complexes as complexes in filtered
modules; see *Algebraic Geometry over the Complex Numbers*, §8.4, pp. 58–59
([author's text](https://www.math.purdue.edu/~arapura/algebra/homological.pdf)).
The implementation uses the equivalent homological convention and makes the
block differential and summand inclusions explicit.
