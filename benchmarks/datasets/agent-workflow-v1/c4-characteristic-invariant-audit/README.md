# C4 characteristic invariant audit

This Assurance benchmark is derived from google-deepmind/formal-conjectures
issue 4423 at revision `b2e608fc52d765510915a244bb69b1a2741acc3c`.
It tests whether an agent can distinguish a Boolean C4-free characteristic
from a count of induced four-cycles.

The task requires three independently chosen connected simple graphs that
separate zero count from characteristic value, multiplicity from a Boolean
invariant, and induced from non-induced cycle detection. The verifier
enumerates every four-vertex subset and all cyclic orders; no graph result is
accepted by label or answer matching.

Difficulty is **Hard (provisional)** because the task combines source-semantic
diagnosis with three constrained graph constructions and exact invariant
replay. Empirical baseline calibration is not yet available.
