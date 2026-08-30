# Projective plane-curve singularity profiles

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`algebraic_geometry.projective_plane_curve.singularity_profile.compute`
computes the complete projective singular locus of one bounded homogeneous
polynomial `F` in `QQ[X,Y,Z]`. The geometric scope is the algebraic closure of
`QQ`; this is not a search for rational singular points.

The returned profile retains the primitive positive-leading integer scalar
representative of `F`, the ordered projective axis, all three exact partial
derivatives, and the saturated homogeneous Jacobian ideal

```text
<F, F_X, F_Y, F_Z> : <X,Y,Z>^infinity.
```

It then has one mathematical outcome:

- `SMOOTH_OVER_ALGEBRAIC_CLOSURE` retains the unit ideal `<1>`, whose
  projective zero locus is empty.
- `SINGULAR_ZERO_DIMENSIONAL` returns every geometric point in a disjoint
  three-chart cover. Each point has a canonical first-nonzero-coordinate
  normalization, one exact presented number field, one indexed real or complex
  embedding, and an exact zero first jet.
- `SINGULAR_POSITIVE_DIMENSIONAL` retains the exact saturated ideal,
  projective dimension one, and its complete family of rational minimal
  components. The saturated ideal remains the authoritative geometric-locus
  evidence; the rational components are not advertised as an absolute
  decomposition.

Backend unavailability, cancellation, timeout, malformed output, and output
limit exhaustion are separate typed operational outcomes. None implies
smoothness or absence of a singular point.

## Exact finite-point construction

The finite branch uses the disjoint charts `X=1`, then `X=0,Y=1`, then
`X=Y=0,Z=1`. Singular 4.4 computes the projective saturation and the rational
minimal primes of each retained chart. A zero-dimensional rational prime is a
number-field residue algebra. A deterministic separating form gives a
univariate irreducible presentation and reduced power-basis expressions for
the remaining coordinates. Enumerating that presentation's exact embeddings
then enumerates every geometric point in the prime's Galois orbit.

For example, the curve

```text
Z (X^2 + Y^2 + Z^2) = 0
```

returns two distinct points with the shared presentation
`QQ(alpha)=QQ[x]/(x^2+1)`: `[1:alpha:0]` under root indexes zero and one. Thus
`[1:-i:0]` and `[1:i:0]` survive strict JSON serialization as different
embedded points.

The point worker constructs reduced coordinates from the exact disjoint-chart
components and their irreducible residue-field presentations. Direct
defining-invariant tests independently substitute those coordinates into `F`
and every partial and reduce modulo the presentation; the execution path does
not replay that computation as a verifier. Numerical matching and backend root
objects do not cross the public boundary.

## Execution envelope

The first envelope admits nonzero homogeneous ternary polynomials of degree one
through three, at most ten sparse terms, at most sixteen digits per raw rational
coefficient component, and at most eight digits after primitive integer scalar
normalization. The degree bound is tied to the complete finite-result
obligation: Bézout bounds the zero-dimensional Jacobian locus by
`(degree-1)^2 <= 4`, matching the degree-four residue-field and four-point
carriers.

Admission derives one execution plan before launching Singular. In the largest
cubic case it charges four Jacobian generators, at most forty source/partial
terms, a fifteen-monomial homogeneous Macaulay layer, coefficient growth from a
Hadamard minor bound followed by Landau–Mignotte factor growth, quotient degree
four, at most seven separating-form attempts, and the retained exact result.
The derived coordinate components remain below the number-field carrier's
256-digit limit. Every decoded ideal is checked against the same plan before it
can enter point construction or a result: at most 64 generators and 1,024
aggregate terms, generator degree at most four, and rational coefficient
components no wider than the derived Macaulay-minor bound. Admission prices the
largest retained saturation and component family from those limits, plus four
embedded-point records and fixed JSON framing. A maximal-shape canonical
serialization remains below the 10-MiB transport boundary; an output that
contradicts the plan becomes a stage-specific `LIMIT_EXCEEDED` outcome.

All Singular calls and exact point construction share one request-scoped
deadline. Singular and the one-shot SymPy point worker are killable and have
fixed diagnostic, address-space, file-size, and exact-output limits. Singular
is the private exact ideal engine; the isolated SymPy transaction supplies
factorization, Gröbner shape conversion, and residue-field construction. Public
identities use only Jacobian's canonical polynomials, ideals, number fields,
field elements, and embeddings.

The equation is never silently square-free reduced. In characteristic zero a
projective plane curve has a positive-dimensional singular locus exactly when
its defining polynomial has a repeated irreducible component; a reduced plane
curve has only finitely many singular points. This dichotomy selects the result
branch while the exact saturated ideal remains the defining invariant.
