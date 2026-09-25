# Cyclotomic character coordinates and equality

`ModularFormCoordinates` can represent a vector in the canonical q-Sturm RREF
basis returned by `modular_form.character_basis.compute` for the bounded
conductor-13 character family. The coordinate `basis_id` is
`gamma0-cyclotomic-character-sturm-rref-v1`; each entry belongs to the exact
`Q(zeta_6)` coefficient field retained by its `ModularFormSpace`.

The admitted spaces are weight-two `M` and `S` spaces at levels 13, 26, and 39
with an even character of conductor 13 whose values lie in `Q(zeta_6)`. The
dimension is computed by the exact Cohen--Oesterle character-sum formula. A
coordinate vector must have exactly that dimension. In a zero-dimensional
cuspidal space the unique vector is the empty tuple. The original
one-dimensional order-six `S_2(Gamma0(13), chi)` spaces retain their existing
basis identifier and coordinate behavior.

The existing `modular_form.equal.check` operation compares two character
coordinate vectors when they have the identical exact space and canonical
basis identifier. Equality of vectors is equivalent to equality of the
represented forms because the q-Sturm RREF basis is linearly independent and
determining through the Sturm bound. The basis producer checks its dimension
against the independent formula and verifies the full q-Sturm rank before it
publishes the basis. Equality revalidates the exact character parent, basis
identifier, coordinate count, and coefficient field; it does not replay basis
construction or expand q-series.

This equality slice admits levels at most 39, character unit tables of at most
24 entries, basis dimension at most 32, field-element coordinates of at most
256 decimal digits, and exact comparison work at most 1,000,000 units. It
compares coordinates in the same parent only. Cross-level or cross-character
transport, nonidentity coefficient-field maps, Hecke actions on these generic
vectors, and equality of unrelated space presentations remain unsupported.

[Gamma0 basis construction](modular-forms-gamma0-rational-bases.md) ·
[Number-theory operations](index.md)
