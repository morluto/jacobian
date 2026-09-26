# External product of cubical chains

`topology.cubical_chain.external_product.compute` accepts two finite,
homogeneous integral cubical chains and returns their sparse external product.
For canonical cell generators, it forms the Cartesian product cell by
concatenating left-factor axes before right-factor axes; coefficients multiply
and extend bilinearly. The empty term list is the zero chain in its retained
ambient dimension and degree.

The product orientation gives the graded boundary identity

```text
∂(a × b) = (∂a) × b + (-1)^p a × (∂b),  a ∈ C_p.
```

Thus the generator product itself carries no additional sign. The sign appears
when a boundary passes the left factor. This is the standard cubical chain
cross product convention; see [Kaczynski, Mischaikow, and Mrozek, *The Cubical
Cohomology Ring: An Algorithmic Approach*, Proposition 2.1](https://doi.org/10.1007/s10208-012-9138-4).

Each chain retains its ambient dimension and homogeneous degree. Its cells are
unique and sorted by interval tuples, coefficients are nonzero integers, and
the zero chain keeps its axes and degree. Inputs have at most 2,048 terms,
64-digit coordinates, and 128-digit coefficients. Product work is admitted by
the pair count; the result is limited to 2,048 terms and 2 MiB before product
cells are built. Products whose exact integer coefficients exceed 128 digits
are rejected before cell construction.
