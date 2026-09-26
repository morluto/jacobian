# Positive-root height and support profiles

`root_system.positive_root_profile.compute` returns every positive root in the
existing canonical lexicographic order of its simple-root coordinates. Each
row gives the root's height (the sum of those coordinates), the indices of the
nonzero simple-root coordinates, and its irreducible factor index. The result
retains the exact finite Cartan datum, so the coordinate and factor axes survive
serialization.

For each irreducible factor, `highest_root_index` points into that same complete
positive-root axis. It selects the unique root that dominates every positive
root of the factor in the simple-root order: `alpha <= beta` when `beta-alpha`
has nonnegative simple-root coordinates. In a reducible datum there is one
selected highest root per factor. The component factors and their root-index
subsets are ordered canonically by their least simple-root index and by the
root axis, respectively.

The finite Cartan admission supports rank at most 8 and at most 120 positive
roots. Before root enumeration, this operation admits the worst-case
root-dominance comparisons and a 64,000-byte conservative result envelope;
the canonical transport limit is 10 MiB. No partial root family is returned.

[Operation references](../index.md) · [Tool surface](../../tools.md)
