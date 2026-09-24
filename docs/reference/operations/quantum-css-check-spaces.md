# Binary CSS stabilizer check spaces

`quantum.stabilizer.css_check_space.compute` accepts separate binary X- and
Z-check matrices on one ordered qubit register. It row-reduces each family over
GF(2) and checks every cross inner product. If `H_X H_Z^T = 0`, it returns a
successful `CSSCheckSpaceValue` retaining the two canonical, role-labelled
families and their combined canonical `CheckSpaceValue`. The combined value
composes directly with stabilizer syndrome, normalizer, and error-equivalence
operations. Otherwise the result contains an exact input-row pair and row
vectors whose inner product is one.

`quantum.stabilizer.css_logical_frame.compute` consumes that typed CSS value
and returns `k = n - rank(H_X) - rank(H_Z)` paired logical X/Z representatives.
The X representatives form a deterministic complement of `row(H_X)` in
`ker(H_Z)`; the Z representatives lie in `ker(H_X)` and pair with the X
representatives by the identity matrix. Thus they give a symplectic basis of
`S^perp/S` with separately retained CSS roles. Before using the source value,
the operation rechecks register binding, canonical independent role bases,
`H_X H_Z^T=0`, and equality between the combined check space and the two CSS
row spaces. A malformed or mismatched caller-supplied carrier is rejected.

`quantum.stabilizer.css_distance.compute` exhaustively computes the separate
CSS distances

- `d_X = min wt(x)` for `x ∈ ker(H_Z) \ row(H_X)`;
- `d_Z = min wt(z)` for `z ∈ ker(H_X) \ row(H_Z)`.

It returns one minimum representative per sector, breaking ties by the
lexicographically first tuple of ordered register positions. A code with
`k=0` returns a structural no-logical-qubits result with no distance fields.
The operation searches all binary supports in increasing weight and admits the
complete candidate count and row-check work before enumerating. This exact
search accepts up to 19 qubits when its complete two-sector bound fits the
1,100,000-candidate and 100,000,000-work envelope; larger searches are rejected
before enumeration. The result is an asymmetric X/Z distance pair, not a claim
about the minimum weight of a mixed X/Z logical Pauli.

All three operations admit at most 32 qubits and 64 total check rows.
Pairing, row-reduction, quotient-complement, dual-frame, and distance-search
work are bounded before expansion. They do not choose generator
phases/eigenvalues, build a code-space projector, or decode errors.

The CSS orthogonality condition and X/Z construction follow the standard
stabilizer formalism; see Gottesman, [*Stabilizer Codes and Quantum Error
Correction*](https://arxiv.org/abs/quant-ph/9705052), Chapter 3.
