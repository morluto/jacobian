# RP2 homology lattice certificate

This Hard (provisional) Regression task transforms SageMath's six-vertex
minimal triangulation of the real projective plane into an exact integral
chain-complex certificate.  Its primary objective is algebraic-topology
reasoning: choose a cycle basis, express all triangle boundaries in it, and
establish the index of the boundary lattice.

The task was selected because the current portfolio did not test integral
homology or lattice-index reasoning.  Nearby examples whose answer could be
obtained from a single rank calculation were rejected: torsion forces the
stronger determinant certificate here.  A memorized `H1(RP2)=Z/2` shortcut is
insufficient because the complete basis-dependent matrix must replay.

The verifier proves only the finite combinatorial calculation for the frozen
triangulation.  It does not identify the realization topologically with RP2
and does not run a proof assistant.  Accordingly the ceiling is `COMPUTED`.

Source: SageMath `simplicial_complex_examples.py`, commit
`8ecee59e510093bf96360177c52825b8e0603e59`, GPL-2.0-or-later.
