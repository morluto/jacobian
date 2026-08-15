# Totient-preimage completeness certificate

This public Regression benchmark converts retained conjecture row C-027 into a
closed finite classification problem: determine the complete preimage of 48
under Euler's totient function without assuming a numerical search bound.

Its single primary objective is exact number-theoretic completeness reasoning.
The key finite reduction is independently replayed: every prime divisor must
satisfy `p-1 | 48`, and each prime-power totient contribution must divide 48.
The verifier derives all option branches and compares the accepted set rather
than trusting a submitted cutoff or answer list.

Provisional difficulty is Hard because the response must coordinate the prime
restriction, exponent bounds, a complete 288-branch product, exact
factorizations, and duplicate-free output. The shortcut audit rejects bounded
brute force, answer-only public lists, incomplete prime options, and false
global Carmichael claims. Full reward is only `COMPUTED` for `phi(n)=48`.
