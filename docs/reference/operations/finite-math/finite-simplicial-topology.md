# Finite simplicial topology

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The topology family owns bounded finite simplicial-complex values and their
exact direct operations:

- `topology.simplicial_complex.canonicalize`
- `topology.simplicial_complex.chain_complex.compute`
- `topology.simplicial_homology.compute`
- `topology.simplicial_homology.integral.compute`

The integral operation wraps the same chain-complex-owned `HomologyResult`
returned by `chain_complex.homology.compute`. It retains the canonical `ZZ`
chain value, Smith transformations, source-basis cycles, torsion invariant
factors, and bounding chains. Reduced chains represent the augmentation as the
ordinary differential from degree 0 to a rank-one group in degree -1. The
operation does not create a durable complex, certificate record, or checker
session.
