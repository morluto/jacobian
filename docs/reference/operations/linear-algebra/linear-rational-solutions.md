# Exact rational linear systems

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`linear.rational_solution.compute` returns one exact bounded rational solution
candidate for a supplied system. `linear.rational_inconsistency.compute` returns
one normalized left witness when the supplied system is inconsistent.

Both operations take the entire typed system in their request and return their
candidate or witness inline. They do not consume producer IDs, create evidence
records, or require a checker-installation decision.
