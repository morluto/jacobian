# Weyl-element order

`weyl_group.element.order.compute` returns the exact multiplicative order of
the finite Weyl-group element represented by a bounded word in simple
reflections. The word uses the same left-to-right convention as
`weyl_group.element.length.compute`.

The implementation computes the word's action on every positive and negative
root. This permutation action is faithful: a transformation fixing all roots
fixes the simple-root basis, so it is the identity. The element order is
therefore exactly the least common multiple of the permutation cycle lengths.
The empty word has order 1.

Admission covers finite Cartan recognition, positive-root closure through the
120-root/rank-8 envelope, the 1024-factor word, up to 240 signed roots, at most
`240 * 1024 * 8` coordinate updates, cycle analysis, and the bounded exact
result. The operation returns a mathematical order only
after constructing the complete action; it does not use early stopping or a
partial-search status.
