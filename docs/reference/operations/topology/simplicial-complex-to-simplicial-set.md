# Finite simplicial complex to simplicial set

`topology.simplicial_set.from_simplicial_complex.compute` constructs the
degree-truncated simplicial set associated with a finite abstract simplicial
complex. It takes the canonical complex and a maximum degree `N` and returns a
`FiniteTruncatedSimplicialSet` through degrees `0..N`, together with the source
complex and the index of each source face in the corresponding nondegenerate
target degree that lies within the retained prefix.

In degree `k`, a simplex is a weakly increasing tuple
`(v_0,...,v_k)` whose nonempty support is a face of the source complex. The
face maps delete an entry and the degeneracy maps repeat an entry. Thus each
source face of dimension at most `N` occurs exactly once as a strictly
increasing, nondegenerate tuple; higher-dimensional faces are not claimed by a
prefix with smaller `N`.

The carrier permits at most 32 simplices per degree and 96 total. Before
enumeration, the operation computes the exact number in degree `k` as

```text
sum over source faces sigma, dim(sigma) <= k, of binomial(k, dim(sigma)).
```

It also admits all face and degeneracy map entries, the number of rows touched
by simplicial-identity checks, face transport rows, and a conservative
serialized-output bound. An over-limit request is rejected rather than
returning a partial prefix. The checked result composes directly with
`topology.simplicial_set.normalized_chains.compute` and other finite-prefix
operations.

The face and degeneracy conventions follow the simplex-category definitions
in [Stacks Project, Simplicial Sets, §14.11](https://stacks.math.columbia.edu/tag/0174).
