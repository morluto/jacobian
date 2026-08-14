# Finite simplicial topology

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The topology family owns bounded finite simplicial-complex values and their
exact direct operations:

- `topology.simplicial_complex.canonicalize`
- `topology.simplicial_complex.chain_complex.compute`
- `topology.simplicial_homology.compute`
- `topology.simplicial_homology.integral.compute`

The integral homology result includes its returned Smith transformations,
cycles, and torsion data as typed mathematical values. It does not create a
durable complex, certificate record, or checker session.
