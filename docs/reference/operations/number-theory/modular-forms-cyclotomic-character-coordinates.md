# Cyclotomic character-space bases

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

`modular_form.character_basis.compute` accepts an optional coefficient count
`precision`. Omitting it returns the smallest Sturm-determining prefix; a
larger request returns the same canonical basis vectors through that many
coefficients, with row reduction still normalized by every Sturm pivot. The
requested count must be at least the space's Sturm precision and at most 128.
The basis identifier and first Sturm-determining coefficients are therefore
stable across requests. These longer exact prefixes support finite operator
reconstruction while retaining the same space and coefficient-field parent.

The basis producer checks its dimension against the independent formula and
verifies the full q-Sturm rank before it publishes the basis. Beyond the
existing character-specific transport and comparison path, these generalized
bases do not add generic equality, Hecke actions, cross-level transport, or
nonidentity coefficient-field maps.

[Gamma0 basis construction](modular-forms-gamma0-rational-bases.md) ·
[Number-theory operations](index.md)
