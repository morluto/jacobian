# Ratio-test boundary separation

This Assurance benchmark transforms the paired divergent and absolutely convergent ratio-test boundary exercises in TaoAnalysisBench (`taobench_000356_textbook` and `taobench_000357_textbook`) at immutable revision `339937d75342072a31903739b1bbbe72e1b40c21` (CC-BY-4.0).

The agent must certify two outcomes, not merely recall that the ratio test is inconclusive at limit one. The clean-room verifier checks the harmonic dyadic-block lower bound, a telescoping positive series, exact partial sums, and both ratio-error identities. Freely chosen checkpoints prevent answer-file replay.

Family: **Assurance**. Primary objective: **assurance calibration for an inconclusive convergence criterion**. Difficulty: **Hard (provisional)** because the response must coordinate two infinite arguments and exact certificates; baseline calibration may place it at Medium-Hard. The assurance ceiling is `COMPUTED`: symbolic identities and finite checkpoints are replayed, but no proof assistant establishes the quantified limit arguments.

Shortcut audit: a memorized label, a finite decimal trace, or either witness alone fails. Nearby routine series evaluations were rejected because they would add only calculation coverage.
