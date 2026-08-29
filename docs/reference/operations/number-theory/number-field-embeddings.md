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

**Scope:** Part of #1689; unblocks #2870. This operation and its field/element
carriers provide separate serializable embedding identities for `i` and `-i`.
They deliberately do not add the umbrella issue's element-conjugate profile,
Minkowski embedding, order recognition, order-ideal construction or arithmetic,
maximal-order computation, or rational-prime splitting operations.

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
`EmbeddedSimpleNumberFieldElement` retains that abstract element together with
the complete field presentation and one selected exact embedding. It does not
use an isolating rectangle as embedding identity.

## Execution envelope

The canonical field carrier admits degree at most 8 and at most 256 decimal
digits per defining coefficient. Before enumerating roots, admission derives:

- Mignotte separation bounds for distinct roots and for distinct real
  coordinates of nonreal roots;
- the size and coefficient-height envelope of the exact real-coordinate
  elimination resultant;
- rational isolation precision and intermediate component digits;
- a Hadamard bound for the defining-polynomial discriminant;
- exact root-ordering and isolation work; and
- the retained-source JSON result against the 10,485,760-byte canonical
  transport envelope.

The elimination resultant is admitted from a `2n`-square Sylvester determinant
and at most `2n^2 + 1` integer coefficients before it is expanded; its storage
envelope is 2,097,152 bits. Exact root refinement is limited to 32,768 bits per
call. An exact Sturm count supplies the signature after the Sylvester
intermediate is admitted; inputs whose pair-order precision is too large are
rejected at that point. Accepted inputs then receive one exact all-root
isolation pass for deterministic rational evidence candidates. The kernel makes
`3r2` rational refinements: two per conjugate pair for sign and pair order, and
one to bind the pair to its admitted evidence rectangle. The complete JSON byte
estimate counts every repeated presentation, indexed polynomial, and four
rational components per record (the larger record shape) before embedding
construction.

The public pair order is mapped to private backend indexes by exact rational
refinement at the admitted separation precision. SymPy 1.14 supplies maintained
exact polynomial irreducibility, resultants, root isolation, `CRootOf`
refinement, and discriminants. No SymPy expression or floating approximation
appears in a public value.
