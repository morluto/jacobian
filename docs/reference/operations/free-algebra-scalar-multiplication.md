# Free-algebra polynomial scalar multiplication

`free_algebra.polynomial.scalar_multiply.compute` multiplies every
coefficient of a sparse noncommutative polynomial by one exact rational. It
returns the canonical `FreeAlgebraPolynomial` directly, preserving the ordered
alphabet and every word when the scalar is nonzero. A zero scalar or zero
polynomial returns the zero polynomial over the same alphabet.

The operation admits the full polynomial value shape: up to 4,096 terms,
64-letter words, and 64 decimal digits per input coefficient component. Before
coefficient arithmetic, it bounds estimated rational work, exact result
allocation, and intermediate scalar cells. Results retain at most 64 digits
per coefficient component. A nonzero scalar with more than 128 digits is
rejected for a nonzero polynomial: a source coefficient component has at most
64 digits, so cross-cancellation cannot reduce the larger scalar component to
the result's 64-digit ceiling. Zero scalars and zero polynomials are handled
before that growth check.

For example, scaling `3xy - 2x` by `2/3` returns `2xy - 4x/3` in the same
ordered generator alphabet.
