# Discrete Morse chain contractions

`topology.discrete_morse.chain_contraction.compute` returns an integral strong
deformation retraction from the oriented simplicial chain complex to the Morse
chain complex of one supplied acyclic matching. Its exact simplex bases,
critical-cell bases, differentials, inclusion `I`, projection `P`, and
degree-raising homotopy `H` remain bound to the same source complex.

The convention is

```text
P I = id
d I = I d_M
P d = d_M P
id - I P = d H + H d
```

This certifies chain homotopy equivalence at the chain level; equal homology
groups alone would not provide these maps. Each matched pair is cancelled at
its unit simplicial incidence. The kernel admits at most 32 source cells and
14 matched pairs, and bounds exact matrix work, coefficient growth, and map
output before reduction. Larger matching and Morse-complex requests remain
available through their separate operations.

The construction is a finite based-chain-complex instance of algebraic
discrete Morse reduction. See Sköldberg, [*Discrete Morse Theory for Free
Chain Complexes*](https://arxiv.org/abs/cs/0504090), and Forman,
[*Morse Theory for Cell Complexes*](https://doi.org/10.1006/aima.1997.1650).

The output establishes a chain homotopy equivalence of the associated chain
complexes. It does not return a topological collapse sequence or claim a
geometric deformation retraction of the underlying spaces.
