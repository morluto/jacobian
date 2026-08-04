# Repair a valuation quantifier

The frozen proof translates `gcd(a,b,c,d)=1` into: “the minimum of the four
valuations is zero for at least one prime dividing `n`.” Audit this translation.

Write `/app/submission.json` following the supplied schema and one bound JSON
evidence file at `evidence/valuation-audit.json`.

Submit:

1. a valuation countermodel on at least three distinct primes where the weak
   existential condition holds but the represented four integers have gcd
   greater than one; and
2. a repaired valuation system on at least three primes where every prime row
   has minimum zero, maximum `k`, and row sum `3k`, so the represented integers
   have gcd `1`, lcm `n`, and product `n^3`.

Prime rows must be strictly increasing and exponents are ordered as `(a,b,c,d)`.
Use at least two different zero-coordinate positions in the repaired system.
The maximum permitted assurance is `COMPUTED`; do not claim `VERIFIED`.
