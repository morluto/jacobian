# jacobian/periodic-orbit-polynomial-obstruction

Reconstruct the divisibility obstruction showing that a nonconstant integer
polynomial cannot be the fixed-point count of every iterate of a map on the
integers.

This Regression benchmark is derived from `lm-provers/FineProofs-SFT` train row
312 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f`
(Apache-2.0). The verifier independently checks the squarefree Möbius
coefficient pattern and both prime-modulus reductions; it does not trust the
dataset proof.

## Quality and shortcut audit

Quality score: **87/100**. The primary objective is a general orbit-count
divisibility proof, adding dynamical-period decomposition and a two-prime
polynomial obstruction. Difficulty is **Hard (provisional)** because the key
step must connect exact-period counts, Möbius inversion, polynomial congruences,
and an infinite-prime argument.

A finite sample of iterate counts cannot pass. The certificate must give both
modular reductions with exact symbolic coefficient vectors. The verifier trusts
the standard cycle-decomposition divisibility theorem, Euclid's infinitude of
primes, and the polynomial identity lemma; it therefore caps assurance at
`COMPUTED`.
