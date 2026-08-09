# Extension-field polynomial-map fibers

`finite_field.polynomial_map.fibers.compute` exhaustively evaluates one
univariate polynomial map on a bounded quotient field
`F_p[t]/(m(t))`. It validates that `p` is prime and that the supplied monic
modulus is irreducible, then returns a complete fiber histogram, the first
collision when one exists, a permutation decision, and a SHA-256 digest of the
ordered output sequence.

The current contract supports extension degree at most four, field order at
most 20,000, at most 32 nonzero terms, and exponents at most one billion.
Modulus and field-element coordinates are canonical residues in ascending
power order. Inputs are enumerated by base-`p` encoding with the constant
coefficient least significant.

The producer returns `COMPUTED` evidence. The operator-authorized companion
`finite_field.polynomial_map.fibers.verify` independently checks modulus
irreducibility and replays every evaluation using a separate standard-library
implementation. An accepted replay returns `VERIFIED` evidence bound to the
exact field definition, term payload, enumeration convention, fiber summary,
collision witness, and digest.

This bounded capability establishes only the supplied finite map. It does not
prove a parameterized permutation-polynomial theorem or validate an external
algebraic-geometry argument.
