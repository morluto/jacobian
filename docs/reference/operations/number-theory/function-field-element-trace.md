# Exact relative trace of a function-field element

`function_field.element.trace.compute` returns the field trace

```text
Tr_(L/GF(p)(x))(a) in GF(p)(x)
```

for an element `a` of one admitted finite separable extension
`L = GF(p)(x)[y]/(f)`. The result retains the exact presented extension and
the input element; its value is the canonical reduced rational function in
the existing prime-field rational-function carrier. The degree-one rational
function field uses the identity trace.

For a monic defining polynomial, Newton's identities compute the power sums
`Tr(y^k)` for the basis powers, then linearity gives `Tr(a)`. This avoids root
construction and remains exact in positive characteristic, including when the
extension degree is divisible by the characteristic. Admission bounds every
rational-function numerator and denominator degree, the estimated exact
Newton-sum work, and the serialized result size before those sums are
expanded.

The contract covers the current prime-constant-field carrier and its bounded
single extension. It does not construct an extension tower or a map to a
different base field.

For finite separable extensions, the trace equals the sum over base-field
embeddings; see [Sharifi, *Abstract Algebra*, Chapter 10](https://math.ucla.edu/~sharifi/notes/algebra-ch10.html).

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
