# Complete Galois correspondence for supported quadratic splitting fields

`number_field.galois_correspondence.compute` returns the complete
subgroup/intermediate-field correspondence for an exact splitting field over
`QQ` whose degree is at most two. The degree-one case has one subgroup and one
embedded field. A separable quadratic extension has exactly the trivial and
full subgroups, paired with the extension and the embedded rational field.

The result contains one exact pair per subgroup. Each pair retains the typed
automorphism subgroup, the exact inclusion of its fixed field into the source
extension, the pointwise stabilizer subgroup, subgroup index and order, field
degrees, and whether the subgroup is normal. Two canonical finite-poset values
represent subgroup inclusion and intermediate-field inclusion. Their labels
refer to the same pair axis, so the order-reversing map is explicit and
composes with the existing subgroup/fixed-field and stabilizer operations.

The kernel admits both lattices before construction. The current field and
subgroup carriers permit at most two subgroup nodes and two embedded-field
nodes. Admission bounds automorphism copies, both posets' nodes and relations,
embedding coordinates, and pair values to 32 exact-value allocation units.
The constructor checks the exact identities
`[G:H] = [L^H:QQ]` and `[L:L^H] = |H|`, then checks that fixing the returned
embedded field recovers the paired subgroup. All subgroups are normal in the
supported groups of order one or two.

For `L = QQ(sqrt(2))`, inclusion reversal is
`{1} < Gal(L/QQ)` paired with `QQ < L`. A split quadratic has only `L = QQ` and
the one-element correspondence. This operation does not construct
higher-degree subgroup lattices, normal closures, or intermediate fields.

[Number-theory operations](index.md) · [Tool surface](../../tools.md)
