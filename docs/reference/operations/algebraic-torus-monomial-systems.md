# Homogeneous monomial systems on algebraic tori

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`algebraic_torus.monomial_system.solution_subgroup.compute` returns the exact
subgroup

```text
{x in (C*)^n : product_j x_j^A_ij = 1 for every row i}
```

for one bounded integer exponent matrix. Rows follow `equation_axis`; columns
follow `coordinate_axis`. Signed Laurent exponents are valid because every
coordinate is explicitly nonzero. Systems with zero equations or zero
coordinates retain their declared ambient axes.

## Smith-coordinate convention

The result contains an exact certificate `D = U A V`. Its parameter maps use
columns of the right transformation `V`: if `z` is the ordered Smith parameter
tuple, then

```text
x_i = product_j z_j ^ V[i,j].
```

Combining equations by `U` does not change the solution set. The first
`rank(A)` Smith coordinates obey `z_i ^ d_i = 1`; factors `d_i = 1` are
trivial and are omitted from the compact torsion-character group. A component
label is one canonical residue modulo each remaining factor. The result never
materializes the Cartesian product of labels, so a large finite component
family remains compact. The final `n-rank(A)` columns of `V` give the free
complex-torus map.

`reduced_free_exponent_map` is an exact LLL-reduced presentation of the same
free subtorus. SymPy reduces row bases and returns `B_reduced = T B_smith`.
Consequently, if `F` is the coordinate-by-Smith-parameter map, the public
coordinate map is `F_reduced = F T^T`.
`smith_free_parameters_from_reduced` stores this `T^T` change of parameters;
it is unimodular and makes the reduced presentation directly composable with
later Laurent-polynomial substitutions.

## Scope and bounds

The operation admits at most 16 equations and 16 coordinates, with at most 32
decimal digits per exponent. Those are the inherited transformation-certified
Smith limits. The component count is bounded before execution by Hadamard's
maximal-minor estimate. Its JSON representation is a canonical decimal integer
string. This specifies the transport encoding, not a requirement to compute
with strings or retain strings in migrated native models; see the
[native integer codec requirements](../value-interoperability.md#requirements-for-a-native-integer-codec).
The kernel uses SymPy 1.14's exact `smith_normal_decomp` and
`DomainMatrix.lll_transform`; it verifies the Smith relation, unimodularity,
torsion congruences, free-kernel equations, and basis transport once before
trusted result construction. Public result decoding checks canonical shapes,
axes, conventions, and source binding without replaying those computations.

This contract is only for the toric locus `(C*)^n` and right-hand side one.
General `x^A=b` requires a separately bound consistency and coefficient
representation. Affine branches with zero coordinates are a different
decomposition problem and are intentionally excluded.

The Smith decomposition and global monomial component parametrization follow
Chen and Mehta, [*Parallel degree computation for solution space of binomial
systems*](https://arxiv.org/abs/1501.02237), especially Propositions 1 and 2.
The separation from affine zero-coordinate branches follows Adrovic and
Verschelde, [*Computing all Affine Solution Sets of Binomial
Systems*](https://arxiv.org/abs/1405.0320).
