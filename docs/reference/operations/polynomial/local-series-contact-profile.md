# Pairwise contact profiles of finite Puiseux prefixes

`local_series.puiseux.contact_profile.compute` compares a bounded family of
`TruncatedPuiseuxWindow` values at one center and on one shared known exponent
window. For each pair it returns the least rational exponent where the exact
coefficients differ. If the retained coefficients agree through the common
exclusive precision, the pair is `UNRESOLVED`; finite prefix agreement never
establishes equality of the infinite series.

This is a profile of the supplied formal values. It does not check that the
values solve a polynomial, prove that they are local branches, or assert that
the supplied family contains every branch. For actual local curve branches,
the order of the difference gives their contact exponent once branch prefixes
have been established; see [Contact of two branches, *Singular Points of Plane
Curves*](https://www.cambridge.org/core/books/abs/singular-points-of-plane-curves/contact-of-two-branches/64F9A7C54E11146D06E98B0246838B9C).

The request admits 2 through 32 prefixes, at most 8,192 aggregate support
terms, and at most one million pairwise support visits. The result retains all
source prefixes and is preflighted against the canonical output-size limit.
Different minimal ramification indices are allowed because each exponent is
compared as an exact rational number. Variable, center, lower exponent bound,
and exclusive precision must agree.
