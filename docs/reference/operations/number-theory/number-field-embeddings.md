# Simple number-field embeddings

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`number_field.embeddings.compute` returns every Archimedean embedding of one
bounded presentation

```text
QQ(alpha) = QQ[x] / (f)
```

where `f` is primitive, irreducible in `ZZ[x]`, and has positive leading
coefficient. Degree one is the presentation of `QQ`. The operation returns the
field signature, every exact selected root with rational isolation evidence,
the complex-conjugate grouping, and the discriminant of `f`. This last value is
the defining-polynomial discriminant; it is not a claim about the maximal order
or the field discriminant.

Parsing the carrier establishes only its bounded primitive, positive-leading
canonical shape. The operation recognizes irreducibility inside its isolated
worker; ordinary request or result deserialization never factors the polynomial
or isolates its roots.

**Scope:** Part of #1689; unblocks #2870. This operation and its field/element
carriers provide separate serializable embedding identities for `i` and `-i`.
The adjacent `number_field.real_embedding.element_order.compare` operation adds
exact comparison of two elements at one selected real record. They deliberately
do not add the umbrella issue's element-conjugate profile, Minkowski embedding,
order-ideal construction, general field arithmetic, maximal-order computation,
or rational-prime splitting operations.

## Exact identity and order

A real root is identified by its primitive minimal polynomial and its index
among the increasing real roots. A nonreal root is identified by the same
polynomial and one global index under this order:

1. all real roots increasingly;
2. conjugate pairs ordered lexicographically by the positive-imaginary
   representative `(Re(z), Im(z))`; and
3. within each pair, the negative-imaginary root followed by the
   positive-imaginary root.

Rational intervals and closed rectangles certify those identities, but they do
not define new algebraic values: many valid isolators can describe the same
indexed root. The operation derives each evidence cell from a certified error
box strictly inside the public interval or closed rectangle. Mignotte separation
then proves that the cell contains exactly one root. This avoids ambiguity when
a raw backend isolator happens to place a root on its boundary.

`SimpleNumberFieldElement` stores exactly degree-many reduced rational
coordinates in the ascending basis `1, alpha, ..., alpha^(n-1)`.

`SimpleNumberFieldRealEmbeddingBinding` structurally binds that element to a
complete `RealNumberFieldEmbeddingRecord`. Construction or deserialization
checks only that both values use the same presentation; it does **not** claim
that the defining polynomial is irreducible, that the indexed root is real, or
that the isolation interval is genuine. This distinction keeps semantic root
recognition out of Pydantic validation.

`number_field.real_embedding.element_order.compare` is a theorem-bearing
consumer of that binding. It recomputes `number_field.embeddings.compute`
inside the same request and requires the selected record to equal one exact
record from that complete profile. Only then does it construct SymPy's exact
`QQ(alpha)` value from canonical integer and rational objects. It subtracts in
the quotient field, derives the selected image's primitive minimal polynomial,
identifies its exact real root, and returns `LT`, `EQ`, or `GT` with a closed
rational enclosure of `left - right`. A zero reduced quotient-field difference
establishes `EQ` independently of the selected real embedding.

The same serialized `SimpleNumberFieldPresentation` is the `field` input to
`number_field.discriminant.compute`; no caller-selected polynomial variable or
shape conversion is needed. The native `discriminant(field)` and
`ring_of_integers(field)` functions accept that value directly. Their private
symbol is always `alpha`. For a nonmonic defining polynomial with leading
coefficient `a`, maximal-order computation uses the integral generator
`beta = a*alpha` and substitutes the resulting basis back into the defining
`alpha` power basis.

## Execution envelope

The canonical field carrier admits degree at most 31, preserving the existing
field-discriminant envelope, and at most 256 decimal digits per defining
coefficient. This complete-profile operation separately admits degree at most
8. Before enumerating roots, admission derives:

- a Mignotte separation bound for distinct roots;
- the size and coefficient-height envelope of the exact real-coordinate
  elimination resultant;
- rational isolation precision and intermediate component digits;
- a Hadamard bound for the defining-polynomial discriminant;
- the retained-source JSON result against the 10,485,760-byte canonical
  transport envelope; and
- a separate byte bound for the worker's compact root-evidence projection.

The elimination resultant is admitted from a `2n`-square Sylvester determinant
and at most `2n^2 + 1` integer coefficients before it is expanded; its storage
envelope is 2,097,152 bits. Exact root refinement is limited to 32,768 bits per
call. An exact Sturm count supplies the signature after the Sylvester
intermediate is admitted. One request-scoped, disposable worker then recognizes
irreducibility, obtains the exact signature by a Sturm count, and expands the
real-coordinate resultant only when more than one conjugate pair needs ordering.
That resultant gives the additional separation bound for distinct real
coordinates. Inputs whose resulting pair-order precision is too large are
rejected before root isolation. Accepted inputs receive exactly one exact
all-root isolation pass at a precision that simultaneously separates roots and
orders positive representatives. The worker coarsens those fine boxes onto the
admitted public evidence grid, so high internal pair-order precision cannot
enlarge a result beyond the component estimate. The complete-result estimate
counts every repeated presentation, indexed polynomial, and four rational
components per record (the larger record shape) before embedding construction;
the worker stdout limit uses the smaller projection estimate instead.

The worker shares the operation's request deadline, is killed on deadline or
client cancellation, and exits after its one response; SymPy root caches cannot
survive into a later request. SymPy 1.14 supplies maintained exact polynomial
irreducibility, resultants, Sturm counts, root isolation, and discriminants. No
SymPy expression or floating approximation appears in a public value.

Selected-element comparison preflights the reduced difference before it repeats
embedding recognition. For `H(alpha)/D`, it bounds the coefficients of
`Res_x(f(x), D*y-H(x))` by a Sylvester-determinant estimate and applies a
Landau--Mignotte factor bound to the selected image's minimal polynomial. The
operation rejects requests whose predicted polynomial exceeds the shared
1,000-digit real-algebraic envelope, whose resultant needs more than 262,144
bits, whose exact root refinement needs more than 32,768 bits, or whose
rational evidence components can exceed 4,096 digits.
