# Coin-process potential certificate

This Hard (provisional) Regression task transforms the Bank of Bath process
into a verifier-first potential-method benchmark.  The primary objective is to
discover a compact invariant that simultaneously proves termination and yields
the exact mean stopping time.

It was selected over a simple simulation benchmark because the submitted
quadratic potential must decrease by one on every one of 4095 nonterminal
states, remain positive, attain certified layer minima, and average correctly.
Knowing the public answer `n(n+1)/4` does not reveal the required coefficients
or satisfy the transition replay.

The verifier exhausts the frozen 12-coin state space with exact integers.  It
does not prove the theorem for arbitrary `n` or invoke Lean, so the assurance
ceiling is `COMPUTED`.

Source: AI-MO/CombiBench row 94, revision
`882ba08befd0856f5364db1e53d58c7e2cf704f9`, MIT license.
