# Rank-one determinant limit

This Regression benchmark converts BeyondAIME row 1 into an exact symbolic
certificate rather than answer recovery. It exercises low-rank matrix
recognition, determinant reduction, telescoping sums, and asymptotic control.

The verifier accepts arbitrary distinct sample sizes and recomputes all
products, sums, polynomial coefficients, roots, and the general positive tail
gap. Difficulty is provisionally Hard. The shortcut audit rejects merely
returning the public answer and floating-point sampling; a complete compatible
certificate is required. Assurance is `COMPUTED` because the verifier replays
the declared algebra but is not a general proof assistant.
