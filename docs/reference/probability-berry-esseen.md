# Berry--Esseen operation

`probability.finite_distribution.berry_esseen_iid_05600.compute` implements the
i.i.d. specialization of Shevtsova's general-independent finite-sum theorem
with universal constant

\[
 C = 0.5600 = 14/25.
\]

The operation's admission envelope is at most 16,384 atoms, 128 decimal
digits in each input rational numerator or denominator, and 512 decimal digits
in every intermediate or result rational numerator or denominator. The sample
count is a positive exact integer whose JSON field uses the same 512-digit
representable envelope; admission of a particular \(n\) uses the reduced
height of the scale-invariant ratio \(C^2\rho^2/(\sigma^6 n)\) rather than an
independent cutoff on \(n\) itself. Affine rescaling that cancels in that
ratio is not charged as unreduced \(\sigma^6\) height.

For one finite rational law \(X\) with positive variance and a positive i.i.d.
sample count \(n\), the operation returns exact rational values

\[
 \mu = E[X],\qquad
 \sigma^2 = \operatorname{Var}(X),\qquad
 \rho = E[|X-E[X]|^3],
\]

and the theorem's explicit Kolmogorov-distance upper bound
\(C\rho/(\sigma^3\sqrt n)\). The square of the bound is returned exactly;
its square root is returned as a deterministic outward-rounded interval with
endpoints on the \(2^{-p}\) grid, where \(p=\texttt{bound\_precision\_bits}\).
A non-singleton interval uses consecutive dyadic endpoints. When the square
root is itself a dyadic of that scale, the interval may be the singleton
\([q,q]\).

The pinned source is I. G. Shevtsova, “An Improvement of Convergence Rate
Estimates in the Lyapunov Theorem,” *Doklady Mathematics* 82(3) (2010),
862–864, DOI
[`10.1134/S1064562410060062`](https://doi.org/10.1134/S1064562410060062).
Here `C=0.5600` is the general-independent theorem's constant, used as a
valid (conservative) specialization for the compact i.i.d. request; the result
does not claim that `0.5600` is an i.i.d.-specific sharp constant.
The operation does not claim to compute the realized distributional distance
or to prove a sharper constant for a narrower hypothesis class.
