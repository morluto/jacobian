Construct a strictly positive continuous function on `[1,+infinity)` for which
the improper integral diverges but the series of integer samples converges.

Use the baseline `b(x)=x^-2`. For each integer `n>=1`, add a height-one
triangular spike centered at `n+1/2`, with half-width `alpha/n`, where you choose
any canonical rational `0 < alpha <= 1/4`. Submit the exact first twelve spike
supports and areas, the twelve integer samples, and the general symbolic series
classifications.

Your certificate must show that the supports are disjoint and avoid every
integer, the spike areas form `alpha * sum(1/n)`, and the samples remain
`sum(1/n^2)`. Write `/app/submission.json` and digest-bound
`/app/evidence/answer.txt`. Claim at most `COMPUTED`.
