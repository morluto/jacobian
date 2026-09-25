# Proper hypergeometric terms

`ProperHypergeometricTerm` is an exact, serialized carrier for the classical
bivariate term

\[
T(n,k)=P(n,k)\,a^n b^k\prod_i(A_i n+B_i k+C_i)!^{m_i},
\]

where `P` is a polynomial over `QQ`, `a` and `b` are nonzero rationals, the
affine coefficients and offsets are integers, and each nonzero signed
multiplicity `m_i` is stored once per affine argument. The polynomial axes are
fixed in the order `(n,k)`.

Positive factorial powers are numerator factors: the term is defined only
where each such factorial argument is nonnegative. Negative powers are
reciprocal factorials, extended by zero at negative integer arguments. Their
nonnegative-argument inequalities define the support. The support and the
region where the term is defined are distinct and remain derivable from the
stored factors. A summation operation must provide its bounds and establish
that every required term value is defined.

The carrier intentionally uses the polynomial prefactor in the classical
proper-hypergeometric definition. An arbitrary rational prefactor can leave
that class and needs a separately specified pole and boundary contract.
Likewise, this value does not claim that a sum has finite support or that a
telescoper exists for a particular summation domain.

The `ore.proper_hypergeometric.shift_quotients.compute` operation returns the
reduced elements of `QQ(n,k)` representing `T(n+1,k)/T(n,k)` and
`T(n,k+1)/T(n,k)`. These are formal identities on the generic locus where the
unshifted term is nonzero and the factorial expressions are defined. They do
not define pointwise division at zeros, poles, or support boundaries; a later
finite-sum argument must use the stored support and an explicit boundary
convention. Both shifted prefactors and factorial products are admitted before
expansion, and each returned rational function is bounded by the shared exact
polynomial carrier.

This structural form is the one used in Wilf and Zeilberger's theorem on
proper-hypergeometric multisums; their paper separately defines support and
well-definedness regions, then derives certificates as rational multiples of
the summand. See [Algorithmic proof theory for hypergeometric (ordinary and
q-) multisum/integral identities](https://sites.math.rutgers.edu/~zeilberg/mamarimY/Zeilberger_y1992_p575.pdf),
especially Sections 3–4. The classical certificate identity is also described
in [Zeilberger, The Method of Creative Telescoping (1991)](https://sites.math.rutgers.edu/~zeilberg/mamarimY/Zeilberger_y1991_p195.pdf).
