# Stabilizer logical Pauli quotient space

`quantum.stabilizer.logical_pauli_space.compute` accepts a register-bound
isotropic phase-free check space `S` and returns the exact quotient
`S^perp / S` over `GF(2)`. Its `LogicalPauliSpace` value retains the canonical
check space, an ordered normalizer embedding into register-bound symplectic
coordinates, and the named quotient coordinate axis. The embedding and section
compose to the ambient Pauli representatives of the quotient basis.

The value includes source-bound linear maps for the stabilizer inclusion into
the normalizer, the normalizer's ambient Pauli representation, the quotient
projection, and a linear section selecting the quotient representatives.
Their axes distinguish check, normalizer, quotient, and ambient symplectic
coordinates. The induced form is a `GF(2)` axis-bound matrix on the quotient
coordinate axis. The quotient dimension is
`2 * logical_qubits`; it is not a stabilizer-group enumeration or a selected
phase lift.

The check rows are validated for isotropy before elimination. Work is admitted
from register width, source row count, and the resulting dense GF(2) map
envelope before constructing the normalizer or quotient bases. The current
carrier admits at most 32 qubits and 64 check rows. Output growth is bounded
before the exact maps and form are allocated. Empty check spaces and
zero-dimensional logical quotients are represented by empty axes and matrices
with their source and target parents retained.

The symplectic form is induced from
`< (x|z), (x'|z') > = x.z' + z.x' mod 2`. Because `S` is isotropic, the pairing
descends to `S^perp/S`; nondegeneracy gives dimension `2(n-r)` for `r`
independent checks. This extends the frame representation of the logical
quotient by retaining its full coordinate projection and section.
