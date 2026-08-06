# Dead-end local-density audit

This task is derived from the MIT-licensed AxiomMath `dead-ends` formalization,
frozen at commit `80fc9124841a1f37a167d227d00780479d04f701` and catalogued in
the maintained Resources spreadsheet at row 239.

The benchmark isolates a proof-critical local factor used in the square-free
digit-walk density argument. It requires separate treatment of primes for which
the base is invertible, divisible once, or divisible by the prime square.
The verifier enumerates every residue independently and checks the submitted
branch classification, obstruction set, count, and reduced density.

It does not replay Lean, certify the infinite Euler product, or establish the
global asymptotic-density theorem.
