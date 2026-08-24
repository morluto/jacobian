# Exact symbolic matrix products

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`matrix.symbolic.multiply.compute` returns the exact row-by-column product of
two bounded matrices over one explicitly ordered field
`QQ(t_1, ..., t_n)`. Entries and the result use the canonical sparse
`RationalFunction` value, so a returned product can be supplied unchanged as
the matrix value to the symbolic rank or linear-system operations.

The request validates equal ordered field variables, compatible dimensions, the
complete unreduced sparse expansion budget, and cancellation-safe per-entry
and aggregate canonical support bounds before execution. The private SymPy
adapter only receives already canonical entries and normalizes every result
entry back to the same rational-function representation.
