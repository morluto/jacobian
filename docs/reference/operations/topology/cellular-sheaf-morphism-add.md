# Add cellular-sheaf morphisms

`cellular_sheaf.morphism.add.compute` adds two natural transformations between
the same exact pair of finite cellular sheaves. It returns the source-bound
component matrices of their pointwise sum over `QQ` or the common `GF(p)`.

Both input maps are re-admitted from their component matrices; serialized
naturality flags are not trusted. Exact equality of source and target sheaves
is required, including their simplicial complex, bases, restrictions, and
coefficient field. Naturality of the result follows because the sum of two
commuting naturality squares commutes. The kernel therefore does not replay the
result's squares a third time.

The operation bounds the canonical component cells and retained result size
before constructing the summed matrices. It rejects a sum whose scalar digits
exceed the existing cellular-sheaf morphism admission envelope, keeping the
result directly usable by other morphism operations.
