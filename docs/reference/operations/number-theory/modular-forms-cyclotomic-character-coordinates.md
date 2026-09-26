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

`modular_form.character_basis.compute` accepts an optional coefficient count
`precision`. Omitting it returns the smallest Sturm-determining prefix; a
larger request returns the same canonical basis vectors through that many
coefficients, with row reduction still normalized by every Sturm pivot. The
requested count must be at least the space's Sturm precision and at most 128.
The basis identifier and first Sturm-determining coefficients are therefore
stable across requests. These longer exact prefixes support finite operator
reconstruction while retaining the same space and coefficient-field parent.

`modular_form.character_coordinates.hecke.apply` also accepts a
one-dimensional character space in this q-Sturm RREF basis. It applies `T_n`
for coprime `n <= 32` when its required source precision
`n * (B - 1) + 1` fits the 128-coefficient basis envelope; here `B` is the
space's Sturm precision. Thus the maximum admitted index can be lower for
spaces with larger Sturm precision (for example, at most 18 when `B = 8`). It
returns coordinates with the identical space and basis identifier: this Hecke
action is an endomorphism, so it does not create an inflated target space.
Along with the source precision, the operation admits work, coefficient growth,
and output size before basis expansion. The transformed form is checked through
the full target Sturm prefix. Multidimensional q-Sturm RREF character spaces
remain unsupported by this Hecke operation.

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
vectors outside the one-dimensional slice above, and equality of unrelated
space presentations remain unsupported.

[Gamma0 basis construction](modular-forms-gamma0-rational-bases.md) ·
[Number-theory operations](index.md)
