# Standard cyclotomic field inclusions

The operations `matrix.cyclic.cyclotomic_inclusion.compute`,
`matrix.cyclic.cyclotomic_inclusion.compose`, and
`matrix.cyclic.cyclotomic_element.map` expose the standard inclusion

```text
QQ(zeta_n) -> QQ(zeta_m),  zeta_n |-> zeta_m^(m/n),  when n | m.
```

The source and target are Jacobian's canonical fields
`QQ[x]/(Phi_n(x))` and `QQ[x]/(Phi_m(x))`, with `x` their named generators.
The inclusion value carries the reduced power-basis coordinates of the image
of the source generator. Application substitutes this image into the source
element and reduces modulo `Phi_m`. Composition accepts only matching
intermediate parents and validates both carried generator images before
returning the canonical direct inclusion. These operations do not describe
arbitrary embeddings or arbitrary isomorphisms between fields that happen to
be equal.

Orders are at most 128. Element coordinates are bounded to 256 decimal digits;
mapping admits field work and a conservative exact numerator/denominator
height bound before substitution. The map result is an ordinary exact field
element and retains its target parent.

The mathematical convention follows the standard cyclotomic construction:
for compatible choices of primitive roots and `n | m`,
`zeta_n = zeta_m^(m/n)`, giving the displayed inclusion. See
[Sharifi, Abstract Algebra, Chapter 6](https://www.math.ucla.edu/~sharifi/notes/algebra-ch06.html)
for the cyclotomic-field and compatible-root conventions.

[Number-theory operations](index.md) · [Operation references](../index.md)
