# Modular-form spaces, bases, and coordinates

`modular_form.space.basis_q_expansions.compute` returns the complete
deterministic basis of a supported rational modular-form space through a finite
precision. For `M_k(SL2Z)`, the basis is the ordered family
`E4^a E6^b` with `4a + 6b = k`. For `S_k(SL2Z)`, it is `Delta` times the
corresponding basis of `M_(k-12)`; it is empty when that inner weight is
negative. This follows the level-one basis construction in
[Stein, *Modular Forms: A Computational Approach*](https://wstein.org/books/modform/modform/weight_two.html).

The operation admits weights through 120 and precisions through 128. It
preflights dimension, convolution work, coefficient growth, and aggregate
output size before constructing any basis coefficient. Each element carries
the same space, finite precision, and basis identifier. Coefficients after
that precision remain unknown.

`modular_form.coordinates.q_expansion.compute` forms any rational linear
combination in that basis. `ModularFormCoordinates` retains the space, basis
version, and exact rational coordinate vector, so it defines a global form,
not only a finite q-prefix.

The basis and coordinate expansion operations return exact finite q-prefixes,
while `modular_form.space.sturm_bound.compute`
returns the exact Sturm integer. Callers can compose these values and compare
the required coefficients themselves; Jacobian publishes no global equality
checker. For a Sturm bound `B`, the determining prefix contains `B + 1`
coefficients, from index zero through index `B`.

These rational basis and coordinate operations support level one, holomorphic
trivial-character spaces at Gamma0(2) and Gamma0(3), even-weight
trivial-character spaces at Gamma0(4), and the represented
`M_1`/`M_3(Gamma0(4), chi_{-4})` spaces, all with rational coefficients.
Beyond those formula families, the bounded PARI Sturm-RREF path in
[Rational Gamma0 modular-form bases](modular-forms-gamma0-rational-bases.md)
supports rational trivial-character `M_k(Gamma0(N))` and `S_k(Gamma0(N))`
spaces through level 10,000 whenever the weight, dimension, Sturm precision,
aggregate work, and output bounds declared by that path admit the request.
Other nontrivial characters remain unsupported by these rational basis and
coordinate operations.

The public Sturm operation has a wider, parent-only contract than the basis
operations: for any represented `Gamma0(N)` space over `QQ` or its declared
rational cyclotomic coefficient field, it returns
`floor(k [SL2(Z):Gamma0(N)] / 12)` and retains the exact character and field in
the result. The formula does not depend on the character or coefficient field;
only the validity of the exact space parent does. A bound `B` determines the
`B + 1` coefficients from q^0 through q^B. PARI documents the same
Gamma0(N), weight-k Sturm bound in `mfsturm(N,k)` in its
[Modular Forms reference](https://pari.math.u-bordeaux.fr/dochtml/html/Modular_forms.html).

`ModularFormSpace` can represent a bounded cyclotomic coefficient parent using
the canonical `RationalCyclotomicField` power-basis value. A narrow
field-valued exception is now supported for even order-6 characters modulo 13:
`modular_form.character_basis.compute` returns a one-element basis of
`S_2(Gamma0(13), chi)` over `Q(zeta_6)`, normalized through q^0..q^2.
`ModularFormCoordinates` represents one exact scalar multiple of that
basis element; `modular_form.character_coordinates.q_expansion.compute`
returns its exact field-valued Sturm prefix. For this space the index is 14
and the weight-2 Sturm bound is 2, so the
returned q^0..q^2 prefix is the determining finite projection. Callers perform
coefficient comparisons on returned values themselves.
`modular_form.character_coordinates.hecke.apply`
supports `T_n` for `1 <= n <= 32` with `gcd(n,13)=1`; it extends the private
PARI basis prefix through `q^(2n)`, applies the exact character-valued Hecke
coefficient formula, and reconstructs the image in the same exact coordinate
space through the Sturm bound. Its coefficient-height admission uses the
weight-2 eigenform bound `|a_m| <= sigma_1(m) <= m(m+1)/2` across the entire
requested prefix. No implicit embeddings across characters,
fields, levels, or basis versions are defined.

For rational scalar multiples of the two conjugate character forms, the
operation `modular_form.character_coordinates.product.compute` multiplies
their exact q-prefixes through the target Sturm bound and returns a
field-valued prefix in `S_4(Gamma0(13))` with trivial character over
`Q(zeta_6)`. The inverse-character relation implies rational coefficients,
which the operation checks exactly. Source parents, target precision, work,
the character-worker's full four-digit coordinate envelope, and output size
are admitted before basis expansion. The product currently returns its
determining q-prefix rather than reconstructed field-valued coordinates.

`ModularFormCoordinates` is the shared exact coordinate value for the
currently admitted rational and cyclotomic slices. Its space retains the
coefficient field, and each coordinate must be a scalar in that field.
For trivial-character rational spaces with an already admitted basis,
`modular_form.coordinates.extend_field.compute` embeds coordinates into the
same basis over `Q(zeta_6)`. The resulting value retains the cyclotomic parent and basis identity, and
`modular_form.field_coordinates.q_expansion.compute` returns an exact
field-valued prefix. This scalar-extension slice preserves the underlying
space. Shared storage does not widen rational-only product, transport, frame,
or Hecke operations; those consumers still enforce their own accepted parents.
The conjugate-character product above is a separate bounded operation that
returns its target Sturm prefix.

This exception does not widen the generic rational dimension, basis, frame,
or operator paths. Those paths remain restricted to their documented rational
families and reject a cyclotomic parent when their operation relies on rational
coefficients. The Sturm operation is the exception: it computes its integer
from the space's level and weight and accepts every exact cyclotomic parent
represented by `ModularFormSpace`. A nontrivial character must be supplied
explicitly at the Gamma0 level; callers must use
`dirichlet_character.inflate.compute` before binding
a character of smaller modulus. The character's value order must divide the
declared cyclotomic field order; `QQ` remains valid for rational-valued
characters. Cyclotomic parent order is at most 128 and field degree at most 32,
checked when the space value is constructed, before an expansion operation can
allocate coefficient vectors. Outside the stated level-13 character slice,
this representation adds no basis, membership, or q-expansion claim for those
parents.

For `chi_{-4}(n) = 0` on even `n`, `1` for `n = 1 (mod 4)`, and `-1` for
`n = 3 (mod 4)`, the basis generator is
`G_{1,chi-4} = 1/4 + sum_{n>=1}(sum_{d|n} chi_{-4}(d))q^n`. Zagier gives this
Eisenstein expansion and shows that `theta(q)^2` has the same weight, level,
and character. His index bound makes the space at most one-dimensional; both
forms are nonzero, and their first coefficients give `theta(q)^2 = 4 G`.
Therefore the returned rational basis spans the full space. See
[Zagier, *Elliptic Modular Forms and Their Applications*, §§2.2 and 3.1](https://www.its.caltech.edu/~matilde/Zagier123ModularForms.pdf).

At Gamma0(2), the holomorphic trivial-character basis is the monomial family
`A2^a E4^b` with `2a + 4b = k`, where
`A2 = 2 E2(2 tau) - E2(tau)`. Zagier proves the free-ring description with
generators `A2` and `E4` in [*Introduction to Modular Forms*](https://people.mpim-bonn.mpg.de/zagier/files/scanned/IntroductionToModularForms/fulltext.pdf).

For even weight `2n` at Gamma0(4), the basis is the ordered family
`B4^(n-j) D4^j`, `0 <= j <= n`, with
`A2 = 2 E2(2 tau) - E2(tau)`, `B4 = A2(2 tau)`, and
`D4 = (A2 - B4)/24`. The known Gamma0(2) form `A2` and its degeneracy
`B4` are modular on Gamma0(4), so `D4` is too. Their expansions begin
`B4 = 1 + 24 q^2 + 24 q^4 + ...` and
`D4 = q + 4 q^3 + 6 q^5 + ...`; hence each listed monomial has leading
term `q^j`, proving independence. The group index is 6, it has three cusps
and no elliptic points, so the dimension formula gives `dim M_(2n) = n+1`
(including weight zero, where constants give dimension one); odd weights
vanish because `-I` acts by `(-1)^k`. The monomials are therefore a complete
basis, with no polynomial relation.

`modular_form.coordinates.hecke.apply` applies the classical coefficient
formula for `T_n` to exact level-one coordinates for bounded `n`, to
`M_k(Gamma0(2))` coordinates for odd `n`, and to the represented
even-weight trivial-character `M_k(Gamma0(4))` bases for odd `n`, and to the
represented `M_1`/`M_3(Gamma0(4), chi_-4)` bases for odd `n`. The coefficient
rule is
`b_m = sum_{d | gcd(m,n)} d^(k-1) a_(mn/d^2)` (with exponent `-1` at weight
zero). This includes `b_0 = a_0 sum_{d|n} d^(k-1)`. For chi_-4 spaces it uses
`b_m = sum_{d | gcd(m,n)} chi_-4(d) d^(k-1) a_(mn/d^2)`. The operation admits
source precision, arithmetic work, and rational growth, then reconstructs the
image through the exact Sturm bound. For
`M_4(Gamma0(2))`, `T_3(E4) = 28 E4`. The source-order gate is `n * B + 1`,
where `B` is the Sturm bound.

For the represented trivial-character `M_k(Gamma0(3))` basis, `T_n` is also
available when `gcd(n, 3) = 1`. The action is reconstructed in the same
`A2^a B4star^b S6^c` basis through the level-3 Sturm bound. Exact coefficient
checks include `T_2(A2) = 3 A2` in weight 2 and
`T_2(A2^2) = 9 A2^2`, `T_2(B4star) = 9 B4star` in weight 4. Admission
preflights the required source order `n*floor(k/3)+1`, work, coefficient
growth, and exact reconstruction size.

`modular_form.coordinates.atkin_lehner.apply` applies the normalized slash action
`|_k W_Q` to an exact even-weight `QQ` form with trivial character on
`Gamma0(N)`, where `Q` is an exact divisor of `N` (`gcd(Q,N/Q)=1`). It preserves
the represented space and deterministic basis. A bounded PARI worker applies
the exact transformation to the canonical q-Sturm RREF representative; the
result is reconstructed into Jacobian coordinates through the Sturm bound.
Admission includes backend basis and transform work, coefficient-digit limits,
and output bytes before backend expansion. `Q=1` is the identity. The independent
fixture is the classical Fricke formula `E4|_4 W_2 = 4 E4(2 tau)`, which follows
from `E4(-1/tau)=tau^4 E4(tau)`. The PARI manual documents the exact
`mfatkininit` transformation:
[PARI modular-forms manual](https://pari.math.u-bordeaux.fr/dochtml/html/Modular_forms.html).
Nontrivial characters and odd weights remain outside this operation's current
rational contract.

`modular_form.coordinates.u2.apply` applies
`U_2(sum a_n q^n) = sum a_(2n) q^n` to exact coordinates in
`M_k(Gamma0(2))`, even-weight trivial-character `M_k(Gamma0(4))`, and in
the represented `M_1`/`M_3(Gamma0(4), chi_-4)` bases. In each represented
space the operation reconstructs the image in the same basis through the
exact Sturm bound. For the weight-four Gamma0(2) basis
`(A2^2, E4)`, `U_2(E4) = -10 A2^2 + 11 E4`. The source-order gate is
`2 * B + 1`. See Zagier, [*Modular Forms of One Variable*, section 2.4](https://people.mpim-bonn.mpg.de/zagier/files/tex/UtrechtLectures/UtBook.pdf).

`modular_form.coordinates.v2.apply` sends exact coordinates in
`M_k(SL2Z)` into `M_k(Gamma0(2))`, or from `M_k(Gamma0(2))` into
`M_k(Gamma0(4))`, by `V_2(sum a_n q^n) = sum a_n q^(2n)`. It reconstructs
the exact target coordinates through that target's Sturm bound. For a
level-one source, the target bound is `B_2 = floor(k/4)`; the required source
precision is `floor(B_2/2) + 1`. For a level-two source, the target bound is
`B_4 = floor(k/2)`; the required source precision is
`floor(B_4/2) + 1`. In both cases coefficients at odd powers are zero and the
coefficient at `q^(2m)` is `a_m`. This is the standard level-raising
degeneracy map; the identity
`2 gamma tau = gamma' (2 tau)` for `gamma` in Gamma0(4) gives the same
transformation law directly.

`modular_form.coordinates.v3.apply` sends exact level-one `M_k` coordinates
to the canonical trivial-character `M_k(Gamma0(3))` coordinates using
`V_3(f)(q) = f(q^3)`. The target Sturm bound is
`B_3 = floor(k/3)`; only source coefficients through `floor(B_3/3)` are
needed. The operation is restricted to the represented rational level-one
holomorphic spaces and reconstructs the result through `B_3`.

These exact coordinate operations do not establish membership from caller-made
finite q-prefixes. Level-two `T_n` is supported only for odd `n`; level-two
`U_2` requires the supported holomorphic coordinate basis; `V_2` requires a
level-one or Gamma0(2) source and returns its exact higher-level target
coordinates; `V_3` currently requires a level-one source.

`modular_form.coordinates.operator_image.compute` binds a level-one
coordinate form to an exact `U_p` or `V_p` image for a prime `p`. The carrier
retains the source coordinates, operator, prime, and trivial-character
`Gamma0(p)` codomain. `modular_form.coordinates.operator_image.q_expansion.compute`
evaluates a bounded prefix directly from those coordinates:

- `U_p(sum a_n q^n) = sum a_(pn) q^n`
- `V_p(sum a_n q^n) = sum a_n q^(pn)`

These coefficient conventions are standard; the level-raising map
`f(tau) -> f(p tau)` gives the `Gamma0(p)` codomain for `V_p`. The bound for
`U_p` follows from `T_p = U_p + p^(k-1) V_p` at level one. See Stein's
[Hecke operator definitions](https://www.wstein.org/papers/generating_hecke/generating_hecke/node2.html)
and [higher-level modular forms notes](https://wstein.org/edu/Fall2003/252/lectures/all/252.pdf).
The evaluator admits source precision, exact basis work, and serialized output
size before constructing coefficients. Both operators preserve the source's
holomorphic or cuspidal kind in this level-one slice.

`modular_form.coordinates.v_degeneracy.apply` applies the classical
degeneracy map `V_d(f)(q) = f(q^d)` directly to coordinates in any currently
represented trivial-character `QQ` space `M_k(Gamma0(M))` or
`S_k(Gamma0(M))`, returning coordinates in the same kind of the exact
`Gamma0(Md)` parent. Both basis plans and the combined work, growth, and output
are admitted before either q-expansion basis is materialized. Coordinate
reconstruction uses the target Sturm bound `B`; only `floor(B/d)+1` source
coefficients are needed. For `d=1` this is the identity parent map. Higher-level
PARI basis support remains restricted to the space kinds accepted by that
existing basis path; nontrivial characters are not inferred or transported.

`modular_form.coordinates.product.compute` multiplies two represented forms
over `QQ` with trivial character and returns exact coordinates in the product
space. The target level is `lcm(N_1,N_2)`, the weight is `k_1+k_2`, and the
target is cuspidal when either input is cuspidal. The current product kernel
uses the native integer-coefficient basis family through level four. It builds
both inputs through the target Sturm precision, multiplies their prefixes, and
recovers target coordinates through the same Sturm bound. This uses modular
form closure in the target space; the separate formal q-series transforms
remain finite series operations and do not assert modularity. Unsupported
characters, coefficient fields, target levels, or basis kinds are rejected
before coefficient expansion.


### Gamma0(3) trivial-character spaces

`modular_form.hecke_matrix.compute` returns the exact action matrix of `T_n`
in the represented canonical basis. The result retains the ambient space,
basis identifier, Hecke index, and ordered row and column labels. Its convention
is `entries[row][column]`: that entry is the coefficient of the row-labeled
basis vector in `T_n` applied to the column-labeled vector. Thus multiplying
the matrix by a coordinate column gives the same coordinates as
`modular_form.coordinates.hecke.apply` when that coordinate operation accepts
the space. The matrix operation supports the explicit level-one through
level-four basis families and the rational trivial-character Gamma0 basis
supplied by the bounded PARI path. It accepts only indices coprime to the
level. It generates one basis prefix for all columns and checks every
reconstructed column through the Sturm bound. Source precision, aggregate
work, exact growth, and matrix output are admitted before coefficient
generation. The matrix convention is that each column gives the image of the
corresponding column-labeled basis vector.

For example, the PARI-backed `M_4(Gamma0(5))` basis has three elements and a
Sturm prefix through `q^2`. Its `T_2` matrix is computed exactly from basis
coefficients through `q^4`; its columns agree with the direct coefficient
formula. Higher-level requests are admitted only when the PARI basis, Sturm
source prefix, exact matrix growth, and result bytes fit their declared bounds.
Coordinate `T_n` application remains limited to the explicit level-one through
level-four families.

For `M_k(Gamma0(3))` over `QQ`, the canonical reduced basis uses

```text
A2 = (3 E2(3 tau) - E2(tau)) / 2          weight 2
B4star = (9 E4(3 tau) - E4(tau)) / 8     weight 4
S6 = eta(tau)^6 eta(3 tau)^6              weight 6
```

The graded ring has the relation `B4star^2 = A2^4 - 108 A2*S6`. The
deterministic basis consists of `A2^a B4star^b S6^c` with `b` in `{0,1}`
and `2a + 4b + 6c = k`; this reduced monomial count agrees with the exact
Gamma0(3) dimension formula. The basis is admitted through weight 94, where the
existing 32-coordinate carrier remains sufficient, and through precision 63.
Basis requests at odd weights return the empty basis because the trivial
character space vanishes under `-I`. The coefficient construction uses exact
divisor sums for the weight-2 and weight-4 generators and a finite integer
product for `S6`; the q-prefix retains its `M_k(Gamma0(3))` parent and basis
identity. Precision is capped at 63 terms to bound the eta-product and output
work; weights above 94 exceed the 32-coordinate value envelope.

The generator formulas, relation, and graded-ring statement are recorded in
Brandon Williams, “The Rings of Hilbert Modular Forms for Q(sqrt(29)) and
Q(sqrt(37)),” Section 13, where the classical `Gamma0(3)` ring is used:
<https://math.berkeley.edu/~btw/hilbert5.pdf>. The product `eta(tau)^6 eta(3 tau)^6`
is also the standard normalized newform in `S_6(Gamma0(3))`.

## Exact rational basis frames

`modular_form.basis_frame.create` declares an ordered rational basis relative
to one of the six supported canonical bases. A frame retains its exact modular
space, canonical basis ID, canonical source labels, caller labels, and a square
matrix `C`; column `j` gives the canonical coordinates of caller basis vector
`j`. Thus canonical coordinates `x` convert to caller coordinates by solving
`C y = x`, and convert back by `x = C y`. Frame creation checks matrix
invertibility before publishing; both conversion consumers check it again at
use time, including after JSON serialization. The zero-dimensional supported space has the unique empty
frame and empty coordinate vector.

For example, in `M_4(Gamma0(2))`, the columns `(1,1)` and `(0,1)` declare
`c1=A2^2+E4` and `c2=E4` relative to the canonical `(A2^2,E4)` basis. The
round-trip operation returns the existing `ModularFormCoordinates` value, so
canonical q-expansion and operator consumers compose without a second
coordinate carrier. A frame describes a change of coordinates inside a
represented space; it does not establish modularity of arbitrary q-series.
`modular_form.hecke_matrix.in_frame.compute` returns the exact matrix of `T_n`
in a supplied frame. If `C` has the frame vectors as columns in canonical
coordinates and `M` is the canonical Hecke matrix, the returned matrix is
`C^(-1) M C`; row and column labels retain the caller's frame labels. It uses
the same supported spaces and index envelope as canonical Hecke matrices and
admits the exact inversion, conjugation work, rational growth, and serialized
output before computing. For the level-one weight-12 frame
`(E4^3 + E6^2, E6^2)`, the `T_2` matrix is
`((2622, 1323), (-1146, -597))`. This result can be checked columnwise by
applying the canonical coordinate Hecke operation and converting each image
back into the frame.

Each frame basis vector is a q-expansion belonging to the retained ambient
modular-form space.
