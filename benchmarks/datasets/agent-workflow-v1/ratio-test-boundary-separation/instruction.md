# Separate the two outcomes at the ratio-test boundary

Give two positive rational sequences indexed by `n >= 1` whose consecutive-term ratios both tend to one, while one associated series diverges and the other converges. Supply exact ratio identities, nine dyadic lower-bound blocks for the divergent witness, and at least four freely chosen partial-sum checkpoints for the convergent witness.

The verifier independently evaluates every rational checkpoint and replays the submitted symbolic identities. A conclusion label or numerical sampling alone is insufficient. The evidence must contain exactly one `RESULT_JSON:` line equal to the submitted `result` and explain why a ratio limit equal to one is inconclusive. Do not claim proof-assistant verification.
