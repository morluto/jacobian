# Exact fixed fields in supported quadratic splitting fields

These operations connect an exact subgroup of a supported splitting-field
automorphism group with its embedded fixed field. The current field carrier
supports splitting fields over `QQ` of degree at most two, so its Galois groups
have order at most two. The returned field inclusion is exact power-basis data;
an isomorphic defining polynomial without its map into the extension is not a
fixed-field result.

## Operations

- `number_field.galois.subgroup.compute` admits a finite set of exact field
  automorphisms as a subgroup. It checks parent identity, membership in the
  complete automorphism group, the identity, inverses, and closure under
  composition. The result retains the exact automorphism values.
- `number_field.galois.fixed_field.compute` returns the fixed field and its
  exact inclusion into the source extension.
- `number_field.galois.intermediate_stabilizer.compute` checks an exact
  embedding of an intermediate `QQ` field and returns the automorphisms fixing
  its embedded image pointwise.

For `L = QQ(sqrt(2))`, the trivial subgroup fixes `L`, while its full order-two
automorphism group fixes `QQ`. In the reverse direction, the stabilizer of
embedded `QQ` is the full group and the stabilizer of embedded `L` is trivial.
The operations retain both inclusions and maps, so these round trips are
composable and distinguish an embedded subfield from a merely isomorphic
presentation.

The implementation does not construct higher-degree normal closures,
composita, intersections, or general intermediate fields. A subgroup candidate
whose elements are not a closed subset of the exact parent automorphism group
is rejected before a fixed field is returned.
