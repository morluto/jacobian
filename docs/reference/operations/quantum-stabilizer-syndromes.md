# Binary stabilizer syndromes

`quantum.stabilizer.syndrome.compute` accepts an isotropic check space and a
phase-free Pauli error bound to the same ordered qubit register. It returns one
binary coordinate for each row in the check space's canonical GF(2) RREF basis:

\[
s_i = \langle g_i,e\rangle = x_i\cdot z_e + z_i\cdot x_e \pmod 2.
\]

The result retains both the canonical check axis and error value. A zero
syndrome is exactly the condition that the error lies in the symplectic
orthogonal \(S^\perp\), the phase-free stabilizer normalizer. Two errors having
the same syndrome differ by an element of \(S^\perp\); they need not differ by
a stabilizer, so syndrome equality alone does not establish error equivalence
or identify a logical coset.

The operation accepts a dependent isotropic basis presentation and first
reduces its row space to canonical RREF. It rejects non-isotropic authored
spaces, malformed native values, and errors bound to another register. The
current envelope is at most 32 qubits and 64 input check rows; the output
syndrome has at most 64 coordinates. Computation uses bounded GF(2) elimination
and pairings and does not enumerate errors or stabilizer elements.
