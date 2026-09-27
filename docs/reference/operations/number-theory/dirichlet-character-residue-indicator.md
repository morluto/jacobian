# Dirichlet-character residue-class indicator expansion

`dirichlet_character.residue_class_indicator_expansion.compute` returns the
exact expansion of the point indicator at one unit residue `a` in the character
basis of `U_N = (Z/NZ)^*`:

```text
1_{x=a} = 1/phi(N) * sum_chi conjugate(chi(a)) chi(x),  x in U_N.
```

The result lists every character coordinate in lexicographic order. A
coefficient is represented as `zeta_E^e / phi(N)`, where `E` is the common
cyclotomic order of the supplied group and `e` is the exact coefficient
exponent. The assertion is an identity of functions on the unit group; it
makes no assertion about nonunits. The group parent retains the chosen unit
generator coordinates, so coefficients and basis rows compose with the
existing Fourier-matrix operation.

The supplied group decomposition is authenticated before expansion. Work is
bounded by the unit-group size and dual rank, and the encoded result size is
admitted before constructing the character axis. The operation returns only
the requested coefficient vector, not the full character Fourier matrix.
