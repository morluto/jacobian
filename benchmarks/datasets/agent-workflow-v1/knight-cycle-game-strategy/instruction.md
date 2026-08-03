Determine the optimal guarantee in the frozen 20 by 20 stone-placement game,
but do not submit only the public value.

Submit `/app/submission.json` and `/app/evidence/answer.txt`. Your certificate
must include both:

1. Amy's lower strategy: a complete 200-site parity class containing no pair at
   squared distance 5, plus the exact move-count argument guaranteeing her
   100th placement against arbitrary Ben moves.
2. Ben's upper strategy: a partition of all 400 sites into 100 four-site
   cycles in the squared-distance-5 graph. For every cycle, give the two
   opposite pairs. The response to Amy at one vertex is its opposite vertex;
   the other two vertices must both conflict with Amy's red stone.

The verifier accepts alternative valid parity classes and alternative cycle
partitions. It recomputes every distance, partition, response, and count.
Do not claim `VERIFIED`; this is an exact finite strategy-invariant replay.
