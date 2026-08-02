# Convergence-mode separation

This Assurance benchmark is derived from formal-conjectures issue #3473 and the pinned `ErdosProblems/522.lean` blob `442b79dd1c1151740d9ce17551c1c9c9d77f5987`.

It asks for a replayable typewriter-sequence certificate separating convergence in probability from almost-sure convergence on `[0,1)` equipped with Lebesgue measure. The verifier derives dyadic block sizes, probabilities, and probe hit indices using exact rational arithmetic. It does not rely on frozen answer labels.

Difficulty is **Hard (provisional)** because the task combines probability semantics, an infinite block construction, quantitative convergence, and pointwise nonconvergence. No baseline calibration has yet been run; calibration may place it at Medium-Hard.

The verifier exactly replays levels 1 through 8 and finitely many freely chosen rational probes. Those checks validate instances of the submitted dyadic construction and its indexing rule; they do not enumerate every real point or mechanically prove the universal infinite pointwise claim. The general conclusion still depends on the construction argument that every point has one hit and at least one miss per level. The task audits the distinction between convergence modes and does not adjudicate Erdős problem 522.
