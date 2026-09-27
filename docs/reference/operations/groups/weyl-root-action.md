# Weyl action on root-lattice vectors

`weyl_group.element.act_on_root.compute` applies one canonical finite Weyl
element to a `RootLatticeVector` in the same ordered Cartan datum. It returns
the resulting `RootLatticeVector` directly, preserving the simple-root basis
and datum so the image can be passed to another root-lattice operation.

If `M` is the element's matrix on the simple-root basis and `v` is the input
coordinate column, the result is `Mv`. For a simple reflection `s_i`, the
defining formula is

```text
s_i(v) = v - (sum_j A[i,j] v_j) e_i,
```

where `A[i,j] = <alpha_i^vee, alpha_j>`. The input may be any integral vector
in the root lattice; it need not itself be a root. The operation re-admits the
caller-supplied matrix as a Weyl element, checks the exact ordered parent, and
bounds membership work, matrix cells, intermediate coordinate growth, and
result coordinates before constructing the output value.

For `A2`, the simple reflection `s0` sends the first simple root `(1, 0)` to
`(-1, 0)`. In `B2`, the ordered Cartan matrix records unequal root lengths, and
the same formula uses its exact row rather than an assumed symmetric matrix.

This operation applies one supplied element to one typed root-lattice value.
`weyl_group.word.act_on_root_vector.compute` instead accepts a reflection word
and raw coordinates; `root_system.positive_roots.compute` returns the complete
positive-root family.

[Operation references](../index.md) · [Tool surface](../../tools.md)
