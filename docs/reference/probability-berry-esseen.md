# Berry--Esseen operation

`probability.finite_distribution.berry_esseen_05600.compute` implements the
published independent, not-necessarily-identically-distributed finite-sum
form with universal constant

\[
 C = 0.5600 = 14/25.
\]

For independent summands \(X_i\), each with positive variance, the operation
returns exact rational values

\[
 \mu = \sum_i E[X_i],\qquad
 V = \sum_i \operatorname{Var}(X_i),\qquad
 \rho = \sum_i E[|X_i-E[X_i]|^3],
\]

and the theorem's explicit Kolmogorov-distance upper bound
\(C\rho/V^{3/2}\). The square of the bound is returned exactly; its square
root is returned as a deterministic outward-rounded dyadic interval. Repeating
one distribution gives the i.i.d. specialization.

The pinned source is I. G. Shevtsova, “On the asymptotically exact constants
in the Berry–Esseen–Katz inequality,” *Theory of Probability and its
Applications* 55 (2011), 225–252, DOI
[`10.4213/tvp4201`](https://doi.org/10.4213/tvp4201). The operation does not
claim to compute the realized distributional distance or to prove a sharper
constant for a narrower hypothesis class.
