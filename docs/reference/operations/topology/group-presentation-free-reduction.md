# Free reduction of a group-presentation word

[Topology operations](index.md) · [Tool surface](../../tools.md)

`topology.group_presentation.free_reduce.compute` takes a finite generator
count and a word of at most 128 signed generator occurrences. Every generator
index must lie on the supplied axis. The result is the canonical
`FiniteGroupWord` obtained by cancelling adjacent pairs `g g^-1` and
`g^-1 g` until no such pair remains. The empty word is the identity, including
when the generator count is zero.

This is free-group reduction only. It does not apply relators from a group
presentation, decide whether a word is trivial in a presented group, or perform
cyclic reduction or conjugacy simplification. The unique free normal form can
be composed with other operations that consume `FiniteGroupWord` values.

Admission is linear in the input length, with the 128-letter limit checked
before reduction. The output length cannot exceed the input length.

For example, with two generators the input word
`g0 g0^-1 g1` returns `g1`.
