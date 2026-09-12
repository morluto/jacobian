# Berry--Esseen operation

`probability.finite_distribution.berry_esseen_iid_05600.compute` implements the
i.i.d. specialization of Shevtsova's general-independent finite-sum theorem
with universal constant

\[
 C = 0.5600 = 14/25.
\]

For one finite rational law \(X\) with positive variance and a positive i.i.d.
sample count \(n\), the operation returns exact rational values

\[
 \mu = E[X],\qquad
 \sigma^2 = \operatorname{Var}(X),\qquad
 \rho = E[|X-E[X]|^3],
\]

and the theorem's explicit Kolmogorov-distance upper bound
\(C\rho/(\sigma^3\sqrt n)\). The square of the bound is returned exactly;
its square root is returned as a deterministic outward-rounded dyadic interval.

The pinned source is I. G. Shevtsova, “An Improvement of Convergence Rate
Estimates in the Lyapunov Theorem,” *Doklady Mathematics* 82(3) (2010),
862–864, DOI
[`10.1134/S1064562410060062`](https://doi.org/10.1134/S1064562410060062).
Here `C=0.5600` is the general-independent theorem's constant, used as a
valid (conservative) specialization for the compact i.i.d. request; the result
does not claim that `0.5600` is an i.i.d.-specific sharp constant.
The operation does not claim to compute the realized distributional distance
or to prove a sharper constant for a narrower hypothesis class.
