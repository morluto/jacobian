Determine the optimal guarantee in the frozen stone-placement game below,
but do not submit only the public value.

## Complete game rules

The game is played on a 20 by 20 board of sites `(x, y)` with `1 <= x <= 20`
and `1 <= y <= 20`, using 1-based coordinates. The board starts empty. Two
players, Amy and Ben, alternate moves, with Amy moving first. On each move the
acting player places one stone of their color on a previously unoccupied site;
Amy's stones are red and Ben's stones are blue. Amy's placements are
unrestricted, but a red stone may not be placed on a site at squared Euclidean
distance 5 from any existing red stone (a knight move away). Ben's placements
are unrestricted and do not conflict with anything. The game ends when Amy has
no legal red placement available. Amy's score is the number of red stones she
placed; Ben tries to minimize it. The optimal guarantee is the maximum number
`g` such that Amy has a strategy guaranteeing at least `g` red placements
against every Ben strategy, and Ben has a strategy holding her to at most `g`.

## Required certificate

Submit `/app/submission.json` and `/app/evidence/answer.txt`. Your certificate
must include both an independently checkable lower strategy and an
independently checkable upper strategy that together prove the exact optimum:

1. Lower strategy (Amy can guarantee the optimum): a set of mutually
   non-conflicting red-eligible sites and an exact move-count argument showing
   Amy can place the optimum number of red stones against arbitrary Ben moves.
   Any valid independent set plus a correct count argument is accepted;
   a particular parity class is not required.
2. Upper strategy (Ben can hold Amy to the optimum): a partition of all 400
   sites into 100 four-site cycles in the squared-distance-5 graph. For every
   cycle, give the two opposite pairs. The response to Amy at one vertex is its
   opposite vertex; the other two vertices must both conflict with Amy's red
   stone. Any valid cycle partition with these properties is accepted;
   a particular partition is not required.

The verifier recomputes every distance, partition, response, and count. It
accepts any alternative valid lower or upper certificate that proves the same
optimum. Do not claim `VERIFIED`; this is an exact finite strategy-invariant
replay.
