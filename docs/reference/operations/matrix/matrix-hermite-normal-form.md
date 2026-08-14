# Integer matrix Hermite normal form

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`matrix.normal_form.hermite.compute` accepts one bounded integer matrix and
returns its row Hermite normal form (H) and the unimodular transformation
(U) satisfying (H = UA). The full relation is part of the typed result, so
callers can inspect or reuse it immediately.

This is a direct inline computation. It does not materialize the source or
normal form into a server-owned object.
