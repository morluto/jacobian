# Factor and certify reciprocal symmetry

The frozen input gives a degree-16 integer polynomial by coefficients in constant-to-leading order. Recover a complete factorization

`leading_coefficient * product(Phi_order(x)^multiplicity)`

using cyclotomic orders from 2 through 30. Submit the ordered factor list, the independently expanded coefficient vector, its reciprocal vector, `P(1)`, and the reciprocal scalar. Explain why excluding `Phi_1=x-1` corresponds to `P(1) != 0` and why inversion of root-of-unity orbits produces coefficient symmetry.

The verifier reconstructs cyclotomic polynomials and the entire product; coefficient-pattern recognition alone is insufficient. Do not claim `VERIFIED` because no proof assistant certifies the unrestricted source theorem.
