# Balanced row-permutation construction

This Hard (provisional) Regression task turns a constructive matrix theorem
from CombiBench into six replayable balanced matching layers.  The primary
objective is constrained combinatorial construction, not merely checking a
published count.

The current portfolio had graph strategies and finite codes but no edge-
coloring-style decomposition that must respect simultaneous row resources and
symbol quotas.  A shortcut that prints any balanced matrix fails because every
cell is bound to a distinct source position in its original row.  Multiple
valid decompositions are accepted.

The verifier establishes only the frozen finite construction.  It does not
prove the general Brualdi theorem or run Lean, so the assurance ceiling is
`COMPUTED`.

Source: AI-MO/CombiBench row 34, revision
`882ba08befd0856f5364db1e53d58c7e2cf704f9`, MIT license.
