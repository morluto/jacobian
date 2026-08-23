# Powerful-number decision

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`integer.decide.powerful` decides whether every prime divisor of one positive
integer has exponent at least two. It accepts canonical decimal integers from
`1` through the 25-digit execution envelope and returns a source-bound partial
factor certificate. The operation does not compute a complete factorization.

## Exact decision

For the source `n`, the operation derives

```text
B = ceil(n^(1/5)).
```

It strips prime factors in ascending order through `B`. A stripped exponent of
one proves immediately that `n` is not powerful. Otherwise the remaining
cofactor `r` has no prime divisor at most `B`, and `n` is powerful exactly when
`r` is `1` or a nontrivial perfect power.

For the nontrivial direction, suppose a `B`-rough `r > 1` were powerful but not
a perfect power. The gcd of all its prime exponents would be one. It therefore
has at least two prime factors, and the exponent sum is at least five: two
exponents cannot both be two, while three or more exponents sum to at least
six. Every prime factor exceeds `B`, so `r > B^5 >= n >= r`, a contradiction.

The result records one of three exact conclusions:

- `EXPONENT_ONE` includes every factor removed through the first exponent-one
  obstruction. `checked_through` names that prime; the result does not claim
  that the residual is `B`-rough.
- `POWERFUL` includes every factor through `B` and either residual `1` or an
  exact `base^exponent` decomposition of the `B`-rough residual.
- `ROUGH_NOT_PERFECT_POWER` includes every factor through `B` and a `B`-rough
  residual whose exact perfect-power classification is negative.

Every result retains the source, derived cutoff, checked range, removed prime
powers, and residual. Result validation replays the bounded decision, so
changing the source, cutoff, checked range, exponent, residual, conclusion, or
perfect-power witness invalidates the result.

## Execution envelope

The 25-digit cap is a conservative release bound, not part of the mathematical
definition of powerful numbers. It gives `B <= 100000`, hence at most 9,592
trial primes. The source and every intermediate have at most 84 bits; successful
factor divisions total at most 83. A positive perfect-power classification
needs exact roots only for the at most 23 prime exponents through 83. The result
contains at most 42 stripped-factor rows: before an early exponent-one row,
every preceding row consumes at least two source bits; complete branches have
only exponent-at-least-two rows.

Prime generation uses a request-local SymPy 1.14 sieve. Fifth roots, exact roots,
and residual perfect-power classification use python-flint 0.9 integer kernels.
The fifth-root cutoff and every power reconstruction use integer arithmetic;
no floating-point value enters the decision.

The partial-factor criterion and production-scale evidence follow the pinned
[Erdős #366 certificate](https://github.com/techno-optimist/erdos-frontier-atlas/blob/0394e3d3b249439ffabec7d96a3311aa441651b8/certificates/erdos-366/README.md#the-exact-powerfulness-test).
Bernstein's
[perfect-power classification algorithm](https://cr.yp.to/papers/powers-ams.pdf)
shows that the residual subproblem admits an essentially linear-time algorithm.
Jacobian delegates its bounded exact classification to FLINT rather than
porting that algorithm.
