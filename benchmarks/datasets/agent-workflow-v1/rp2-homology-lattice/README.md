# RP2 homology lattice certificate

This Hard (provisional) Regression task transforms SageMath's six-vertex
minimal triangulation of the real projective plane into an exact integral
chain-complex certificate.  Its primary objective is algebraic-topology
reasoning: choose a cycle basis, express all triangle boundaries in it, and
establish the index of the boundary lattice.

This task overlaps with the existing
`public-reproductions-v1/integral-projective-plane` regression, which already
tests unreduced integral homology of a six-vertex RP2 triangulation with
`Z/2` torsion in degree one.  The certificate-level distinction is that this
task requires the complete basis-dependent boundary-coordinate matrix and
its determinant (the lattice index), not just the homology group, so a
memorized `H1(RP2)=Z/2` shortcut or a single rank calculation is
insufficient because the full matrix must replay.  The portfolio gap it fills
is lattice-index reasoning, not first integral-homology coverage.

The verifier proves only the finite combinatorial calculation for the frozen
triangulation.  It does not identify the realization topologically with RP2
and does not run a proof assistant.  Accordingly the ceiling is `COMPUTED`.

Source: SageMath `simplicial_complex_examples.py`, commit
`8ecee59e510093bf96360177c52825b8e0603e59`, GPL-2.0-or-later.
