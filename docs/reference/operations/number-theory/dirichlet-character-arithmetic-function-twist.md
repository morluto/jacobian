# Exact arithmetic-function twist by a Dirichlet character

`arithmetic_function.dirichlet_character_twist.compute` accepts a finite exact
arithmetic-function prefix `a(1), ..., a(M)` from the Dirichlet-convolution
operation and a canonical Dirichlet character `chi`. It returns the exact
finite sequence

\[
b(n)=\chi(n)a(n),\qquad 1\le n\le M.
\]

The result is a `FiniteCyclotomicSequence` with `index_origin=1`. Its single
coefficient field is the character's exact value field; nonunits contribute
zero. This canonical index axis preserves the arithmetic-function convention
and lets the result feed directly into the finite-sequence operations.

This is the finite-prefix coefficientwise transform only. It makes no claim
about convergence or analytic properties of an associated Dirichlet series.
The transform is compatible with Dirichlet convolution when the full exact
functions are defined: twisting both factors and then convolving equals
twisting their convolution. Mathlib records this identity as
[`DirichletCharacter.mul_convolution_distrib`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/LSeries/Dirichlet.html).

The implementation delegates to the shared sequence-twist kernel, which
admits field degree, coefficient growth, work, and serialized output before
evaluating character values. The input arithmetic-function contract bounds
the prefix length; the output inherits the cyclotomic sequence envelope.
