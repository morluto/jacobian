# Additive-group actions from locally nilpotent derivations

`algebraic_group.ga.action_from_derivation.compute` constructs the additive
group action on a bounded rational polynomial ring from a derivation and exact
iterate chains for every generator.

For (R=\mathbb{Q}[x_1,\ldots,x_n]), the request supplies each generator image
\(D(x_i)\) and the chain

\[
x_i,\;D(x_i),\;\ldots,\;D^{N_i}(x_i)=0.
\]

The operation checks each chain transition using the derivation and requires
the final value to be zero. This establishes local nilpotence on the finitely
generated polynomial algebra. It then returns the generator images

\[
\exp(tD)(x_i)=\sum_{k=0}^{N_i-1}\frac{t^kD^k(x_i)}{k!}.
\]

The output polynomial axis includes the source variables and a fresh parameter
axis. If `t` is already a source variable, the operation selects another
canonical fresh name. The returned value retains both axes and the generator
images.

The runtime checks are the certificate replay above, the carrier bounds on
the published action value, and the output preflight below. The operation does
not re-expand the counit \(t=0\) or the two-parameter additive coaction law at
runtime; those identities are structural consequences of the truncated
exponential and are verified against an independent substitution oracle in the
test suite.

The request is bounded to seven source variables so the explicit parameter
fits the polynomial carrier's eight axes. Generator chains contain at most 32
polynomials including the terminal zero. The exponential output is preflighted
to 4096 terms and 2,000,000 estimated expansion cells, and every published
action image must satisfy the source exponent envelope and the 128-digit
coefficient envelope. Source polynomial terms, degrees, and coefficient sizes
use the derivation owner's stated bounds.

The operation handles this `G_a` action construction only. It does not
classify arbitrary group actions, construct `G_m` actions, determine invariant
rings, or infer existence of a certificate from a bounded iterate search.

The native function is
`jacobian.math.polynomials.derivations.ga_action_from_derivation`.

## A supplied finite stable subspace

`algebraic_group.ga.stable_subrepresentation.compute` accepts a checked
`PolynomialGaAction` and an ordered, linearly independent finite list of
polynomials in its source ring. It checks each substituted basis polynomial
and returns the exact action matrix over `QQ[t]`. Matrix columns correspond
to input basis vectors and rows to output basis vectors. The basis order is
part of the value, so its matrix coordinates remain stable across
serialization.

For translation `x -> x+t`, the basis `(1,x)` is stable and the matrix is
`[[1,t],[0,1]]`. A supplied span that does not contain every action image is
rejected. The operation checks only that explicit span; it does not search for
all finite-dimensional subrepresentations. The basis dimension is at most 32,
and degree, term count, expansion work, and coefficient growth are admitted
before substitution.
