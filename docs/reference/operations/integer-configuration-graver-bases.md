# Integer-configuration Graver bases

`integer_configuration.graver_basis.compute` returns the complete sign-normalized
Graver basis in either of two exact slices:

- one-row integer matrices with one to five columns, using the complete
  theorem-bounded coordinate enumeration described by the operation; or
- matrices with at most 12 rows and columns whose integer kernel has nullity at
  most one.

In the second slice, the operation first computes the admitted complete
integer-kernel lattice. A zero-dimensional kernel has empty Graver basis. If
the kernel has rank one, its primitive lattice generator, up to sign, is the
entire Graver basis: every other kernel vector is an integer multiple and has a
conformal decomposition. This avoids any bounded search over candidate vectors.
Matrices outside both slices are rejected; no truncated output is reported as
complete.

The result retains the exact integer matrix and uses its column coordinates.
Each nonzero vector is a primitive relation, sign-normalized so its first
nonzero coordinate is positive. The associated bounded Markov operation uses
this complete Graver basis, which generates the toric ideal and connects every
nonnegative fiber.

The multirow path inherits the relation-lattice operation's admission bounds
of 12 rows and columns and at most eight decimal digits per matrix entry. The
single-row enumeration separately preflights its complete candidate and
candidate-pair work envelopes. This operation does not compute a Graver basis
for a general higher-nullity matrix.
