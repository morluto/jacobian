# Free associative algebra homomorphism composition

`free_algebra.homomorphism.compose.compute` composes two supplied maps in
input order: `f:A→B`, then `g:B→C`, returning `g∘f:A→C`. The intermediate
ordered generator alphabets must agree exactly. Composition returns the
canonical `FreeAlgebraPolynomialHomomorphism` value already used by
substitution and application; it introduces no parallel map representation.

For every generator `x` of `A`, the returned image is the exact polynomial
substitution `g(f(x))`. Thus zero images and noncommuting word order follow the
same admitted kernel as direct map application. Empty source or target axes
remain valid, including the unique map between free algebras on empty axes.

Before expanding any output image, admission accounts for the aggregate
distributive contributions, exact work, coefficient growth, output word size,
and result allocation across every generator image. Each returned image must
also fit the canonical polynomial term and coefficient bounds.
