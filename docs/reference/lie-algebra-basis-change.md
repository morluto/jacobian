# Lie algebra basis transport

`lie_algebra.basis_change.compute` transports a finite-dimensional Lie algebra
over `QQ` to a caller-specified ordered basis. The supplied matrix `P` uses
columns: column `j` is the new basis vector `b'_j` written in the source basis.
The operation returns both coordinate maps: target coordinates to source
coordinates by `P`, and source coordinates to target coordinates by `P^-1`.

For the source bracket and each pair of target basis columns, it computes

```text
[b'_i, b'_j]_source = P [b'_i, b'_j]_target
```

so the target structure constants are `P^-1 [P e_i, P e_j]_source`. This
convention agrees with the usual tensor transformation law for structure
constants under a basis change; see Samelson, *Notes on Lie Algebras*, §1.2
([Cornell-hosted notes](https://pi.math.cornell.edu/~hatcher/Other/Samelson-LieAlg.pdf)).

The exact rational inverse uses FLINT. Admission bounds the source algebra's
Jacobi work, dimension, matrix scalar height, inverse growth, every transformed
bracket intermediate, and result before inversion. The dimension limit is 8;
matrix entries are limited to 128 decimal digits; output structure constants
must fit the existing 64-digit Lie-algebra coefficient contract.

An independent regression realizes `sl2` by `2x2` rational matrices and checks
the transformed brackets there. It also checks both coordinate maps compose to
the identity, along with identity transport and rejection of singular matrices.
