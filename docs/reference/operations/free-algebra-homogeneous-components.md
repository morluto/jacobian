# Homogeneous components of free-algebra polynomials

The native `homogeneous_component` helper projects a sparse
`QQ<X>` polynomial onto one total word degree. The free associative algebra is
graded by word length, so the degree-`d` component is the sum of exactly those
terms whose words contain `d` generator letters. The result retains both the
selected degree and the ordered generator alphabet, even when the projection
is zero.

Terms in Jacobian's canonical polynomial value are already ordered by
descending degree-lexicographic order. Projection preserves their order and
coefficients and creates no new words or coefficient arithmetic.

Degrees are arbitrary nonnegative integers; degrees above the source carrier
maximum return zero. The helper bounds its scan at 600,000 work units. Missing degrees return
the canonical zero polynomial bound to the same alphabet. The requested degree
is preserved exactly on the wire: the JSON encoding is a canonical decimal
string within the shared 32,768-digit exact-integer envelope, so a degree beyond
the interoperable JSON-number range is not rounded.

This helper exposes one graded projection. It does not claim ideal
membership, quotient normal forms, or homogeneous decomposition of a
nonhomogeneous ideal presentation.
