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

Before returning, the producer checks the counit \(t=0\), recovers each
\(D(x_i)\) as the coefficient of (t), and expands the two sides of the
additive coaction law with independent parameters:

\[
(\rho_s\otimes\mathrm{id})\rho_t
= (\mathrm{id}\otimes\Delta)\rho,
\qquad \Delta(t)=s+t.
\]

The request is bounded to seven source variables so the explicit parameter
fits the polynomial carrier's eight axes. Generator chains contain at most 32
polynomials including the terminal zero. The exponential output is limited to
4096 terms and 2,000,000 estimated serialized bytes. The exact two-parameter
coaction check is preflighted to 25,000 expansion cells and 512 coefficient
digits. Source polynomial terms, degrees, and coefficient sizes use the
derivation owner's stated bounds.

The operation handles this `G_a` action construction only. It does not
classify arbitrary group actions, construct `G_m` actions, determine invariant
rings, or infer existence of a certificate from a bounded iterate search.

The native function is
`jacobian.math.polynomials.derivations.ga_action_from_derivation`.
