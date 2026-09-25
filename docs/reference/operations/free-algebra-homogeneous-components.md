# Homogeneous components of free-algebra polynomials

`free_algebra.polynomial.homogeneous_component.compute` projects a sparse
`QQ<X>` polynomial onto one total word degree. The free associative algebra is
graded by word length, so the degree-`d` component is the sum of exactly those
terms whose words contain `d` generator letters. The result retains both the
selected degree and the ordered generator alphabet, even when the projection
is zero.

Terms in Jacobian's canonical polynomial value are already ordered by
descending degree-lexicographic order. Projection preserves their order and
coefficients and creates no new words or coefficient arithmetic.

The operation admits degree values through 64 and at most 600,000 scan/copy
work units. Before building the result it conservatively estimates serialized
output, allowing up to 12 bytes per Unicode scalar for JSON escaping, and limits
output to 2,000,000 cells. Missing degrees return the
canonical zero polynomial bound to the same alphabet.

This operation exposes one graded projection. It does not claim ideal
membership, quotient normal forms, or homogeneous decomposition of a
nonhomogeneous ideal presentation.
