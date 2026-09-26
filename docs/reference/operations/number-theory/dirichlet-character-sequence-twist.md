# Exact finite sequence twist by a Dirichlet character

`sequence.dirichlet_character_twist.compute` accepts an exact finite integer,
rational, or cyclotomic sequence and one canonical Dirichlet character. Integer
and rational inputs require the first index `index_origin`; an existing
cyclotomic input keeps its authored origin. At offset \(j\) it returns

\[
b_j=\chi(\text{index\_origin}+j)a_j.
\]

The output is a `FiniteCyclotomicSequence` with the same index origin and one
declared coefficient field `QQ[zeta_l]`, where `l` is the least common multiple
of the source field order (or 1 for rational inputs) and the character's exact
value order. Nonunits contribute zero. Source and character coefficients are
embedded exactly in that common field, including for empty sequences.

The operation composes: twisting by \(\chi\) and then \(\psi\) gives the same
coefficient values as twisting by \(\chi\psi\), after exact transport to a
common coefficient field. This keeps the original index axis and lets the
returned sequence feed directly into another twist.

An exact arithmetic-function prefix \(a(1),\ldots,a(M)\) given as a rational
sequence with `index_origin=1` is the same postcondition, so the arithmetic
function domain reaches this operation instead of a second declaration. The
native helper
`jacobian.math.number_theory.characters.dirichlet_character_arithmetic_function_twist`
accepts an arithmetic-function value and a canonical character directly and
delegates to the same bounded kernel. When the full exact functions are
defined, the transform is compatible with Dirichlet convolution: twisting both
factors and then convolving equals twisting their convolution. Mathlib records
this identity as
[`DirichletCharacter.mul_convolution_distrib`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/LSeries/Dirichlet.html).
This finite-prefix coefficientwise transform makes no claim about convergence
or analytic properties of an associated Dirichlet series.

Admission bounds the source digit total, character value order, cyclotomic
coefficient growth, coefficient cells, work, and serialized output before
constructing twisted coefficients. The finite sequence length follows the
shared sequence carrier bound; `index_origin` is a signed 32-bit integer.
Modular-form space transport remains owned by the modular-form operations.
