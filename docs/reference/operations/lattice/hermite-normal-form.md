# Integer Hermite normal form

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`lattice.hermite_normal_form.compute` accepts one bounded integer matrix and
returns its row Hermite normal form (H) and the unimodular transformation (U)
satisfying (H = UA). The full relation is part of the typed result, so callers
can inspect or reuse it immediately. Jacobian retains neither input nor result.
