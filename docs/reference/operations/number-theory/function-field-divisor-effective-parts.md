# Function-field divisor effective parts

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.divisor.effective_parts.compute` accepts a finite divisor over
the currently supported prime-constant-field model
`GF(p)(x)`. Its places must pass the same exact field and prime-place admission
used by divisor-degree computation, its finite support is capped at 256 terms,
and each multiplicity is limited to 4096 bits by this operation.

It returns the coefficientwise decomposition

```text
D = D_+ - D_-
```

where both `D_+` and `D_-` are effective and have disjoint support. The
operation does not factor elements or enumerate places; after divisor admission,
it performs a single pass over at most 256 terms and returns no more terms than
the input contains across both parts. At the multiplicity limit, copying the
original divisor and both parts keeps integer payloads below 3 times 256 times
4096 bits, in addition to bounded place and field data. Sorting by structural
place serialization makes the two parts independent of input term order. An
empty divisor maps to two empty effective divisors over the same field.

The general field carrier can represent one finite separable extension with
prime constant field, with extension degree at most 6 and defining coefficient
polynomial degree at most 12, but exact place admission currently supports only
the rational field `GF(p)(x)`. Extension places, extension base fields, towers,
arbitrary divisors from curves, genus, differentials, and complete Riemann–Roch
spaces remain outside this operation's claim.
