# Restricting a cellular sheaf to a subcomplex

`cellular_sheaf.subcomplex.restrict` restricts a checked cellular sheaf to a
finite simplicial subcomplex. The requested complex must be a genuine
subcomplex of the sheaf's complex: every vertex and every nonempty simplex must
already occur in the source face poset.

The result keeps the source sheaf and returns a second `FiniteCellularSheaf`
whose complex is the selected subcomplex. It filters the original stalks and
all comparable-cell restriction matrices, retaining coefficient-field data,
prime modulus, and the exact ordered stalk bases. Since the sheaf diagram is
already checked, restrictions between retained cells are copied directly;
they are not recomputed along new paths. This makes successive restriction
compose: restriction to `L` through an included `K` agrees with direct
restriction to `L` whenever `L` is included in `K` and `K` in the source.

The operation admits at most 64 cells in either complex, 2,048 retained
comparable-cell maps, 65,536 matrix entries, and 8,000,000 serialized result
characters. These limits are checked before the target diagram is assembled.
