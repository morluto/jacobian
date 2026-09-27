# Function-field place uniformizer

`function_field.place.uniformizer.compute` returns an exact element bound to the
requested place and having valuation one there. In `GF(p)(x)`, a finite place
defined by a monic irreducible polynomial `q(x)` uses `q(x)` as its
uniformizer. At the unique infinite place, `1/x` has valuation one. This is the
discrete-valuation-ring definition of a uniformizer: an element generating the
maximal ideal, equivalently an element of valuation one.
These conventions follow the standard treatment of places and local
uniformizers in [Stichtenoth, *Algebraic Function Fields and Codes*](https://link.springer.com/book/10.1007/978-3-540-76878-4).

The operation accepts only the rational function field over a prime field and
validates that each finite place polynomial is irreducible. Its result carries
the canonical place and exact function-field element together, preserving the
parent through serialization. Places on nontrivial extensions, including the
hyperelliptic place carriers, are outside this contract.
