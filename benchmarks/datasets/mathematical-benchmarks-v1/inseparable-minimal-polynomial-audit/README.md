# Purely inseparable minimal-polynomial audit

This Regression-family task freezes DeepTheorem train row 2 (source id 843)
at revision `f5935720f176cedff4ecd8ebf83d1696e31cfac8`, under MIT.

Its single objective is proof diagnosis: distinguish an annihilating
polynomial with repeated roots from the actual minimal polynomial, then repair
the missing irreducibility step through the `u`-adic valuation.  The public
factorization and any tiny characteristic instance cannot pass without the
universal valuation obstruction and minimal-polynomial reconstruction.

Difficulty is **Hard (provisional)**.  The chain crosses transcendence,
rational-function valuations, irreducibility of `X^p-u`, degree, derivative,
and separability.  We expect weaker agents to repeat the source shortcut and
stronger agents to identify and close the logical gap.  Tool-less agents can
solve it, but must sustain the full symbolic chain.

The verifier independently checks the exact certificate relationships and a
freely chosen prime sanity instance.  Its assurance is `COMPUTED`: it is not a
general field implementation or proof-assistant replay.
