# Free-algebra polynomial commutator

`free_algebra.polynomial.commutator.compute` returns the exact associative
commutator `[f,g] = fg - gf` for two sparse `QQ`-linear polynomials over the
same ordered free-generator alphabet. It retains both source polynomials and
returns the canonical sparse result in descending degree-lexicographic order.
The empty support is the zero polynomial with its alphabet preserved.

Admission validates both canonical polynomial values, then bounds the two
convolutions' candidate terms and word cells, exact common-denominator scaling,
coefficient digits, and estimated arithmetic/work before concatenating product
words. The 4,096 candidate-term cap is a conservative upper bound; later
cancellation can reduce the returned support. A resource refusal returns no
partial commutator.
