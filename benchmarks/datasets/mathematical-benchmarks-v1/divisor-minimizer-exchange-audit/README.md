# Divisor minimizer exchange audit

This **Regression** benchmark transforms FineProofs-SFT row 304 at immutable
revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0). It asks for
two consecutive exact divisor minimizers and a certificate explaining their
divisibility, without exposing the source proof route.

The clean-room verifier independently enumerates integer exponent partitions,
assigns decreasing exponents to increasing primes, and proves optimality by
comparing every partition candidate. It also checks the submitted factor maps,
divisor counts, quotient, and a freely ordered complete candidate table.

Family: **Regression**. Primary objective: **exact multiplicative optimization
with an exchange certificate**. Quality score: **87/100**. Difficulty is
**Hard (provisional)** because the agent must discover and coordinate the
partition optimization rather than reproduce a supplied formula. The shortcut
audit rejects a bare pair of integers, a single factorization, incomplete
partition coverage, and any candidate table that omits a competing exponent
shape. Assurance is capped at **COMPUTED**.
