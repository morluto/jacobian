# Exact qubit Pauli label conversion

`quantum.pauli.qubit.from_labels.compute` and
`quantum.pauli.qubit.to_labels.compute` convert between a complete ordered
`I`/`X`/`Y`/`Z` row and the register-bound exact value
`i^r X^x Z^z`. The register order is part of the value, and the row must have
exactly one label for every qubit.

The convention fixes `Y = i X Z`. Therefore a label row with `m` entries equal
to `Y` and scalar phase `s` is represented by `x_q = 1` for `X` or `Y`,
`z_q = 1` for `Z` or `Y`, and stored phase
`r = s + m (mod 4)`. In the reverse direction, labels are recovered from each
coordinate pair and the returned scalar phase is `s = r - m (mod 4)`. This
means the label and phase round trip preserves the exact operator, including
negative and imaginary scalar multiples.

For example, the label `Y` with scalar phase zero has `(x,z)=(1,1)` and stored
phase one. A stored phase of three on the same vector decodes as `-Y`.

Both conversions are linear in register length and preserve the exact ordered
register. They do not expand dense matrices; small dense products are useful
only as independent convention checks. The convention agrees with the
binary-Pauli presentation in [Gottesman's thesis](https://thesis.caltech.edu/2900/).
