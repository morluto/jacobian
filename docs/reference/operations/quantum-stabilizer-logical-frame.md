# General stabilizer logical Pauli frame

`quantum.stabilizer.logical_frame.compute` accepts a register-bound isotropic
phase-free qubit check space `S`, with arbitrary mixed X/Z rows. It computes
the symplectic orthogonal `S^perp` and returns `k = n - rank(S)` pairs of
register-bound representatives whose classes form a symplectic basis of the
quotient `S^perp/S`:

- the X/Z cross-pairing matrix is the `k × k` identity over GF(2);
- pairings within either family vanish; and
- `S` together with the returned representatives spans all of `S^perp`.

Selection is deterministic under the ordered register and canonical check-space
basis, but the representatives are not intrinsic canonical invariants of the
code. The value retains the canonical source `CheckSpaceValue` and the
phase-free representatives, so it preserves register binding and composes
with phase-free Pauli operations. It does not select a stabilizer eigenspace,
lift phases, enumerate the `4^k` quotient, or compute distances.

The operation admits at most 32 qubits and 64 check rows. It constructs a
quotient basis by extending `S` inside `S^perp`, then symplectically
orthogonalizes the complement; it does not enumerate quotient elements. The
logical quotient and its symplectic pairing follow the stabilizer formalism in
[Gottesman's *Stabilizer Codes and Quantum Error Correction*, Chapter 3](https://arxiv.org/abs/quant-ph/9705052).
