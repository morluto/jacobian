# Finite cubical face posets

[Topology operations](index.md) · [Tool surface](../../tools.md)

`topology.cubical_complex.face_poset.compute` closes a finite family of
elementary integer-lattice cubes under faces, then returns the poset of those
cells ordered by inclusion. The target is the existing `FinitePoset` value,
and the result retains `CubicalComplex` plus one mapping entry per canonical
cell. Each entry gives the poset label, source cubical cell, and its cubical
dimension. The face-poset labels are local names (`c00`, `c01`, ...); the
mapping is required to interpret them as cells.

The strict order is `Q < R` exactly when every interval of `Q` is contained in
the corresponding interval of `R` and the cells differ. The cover relations
are the codimension-one cubical faces. In a pure complex the finite-poset rank
agrees with cell dimension; for nonpure complexes, the explicit mapping keeps
cell dimensions even though the poset need not be graded.

Admission uses the existing 64-element finite-poset cap. More than 64 distinct
source cells cannot fit. A dimension-four cube already has 81 distinct faces,
so cells of dimension four or higher are rejected before face expansion.
For admitted source rows, at most `64 * 3^3 = 1,728` candidate faces are
considered; complete output closure is capped at 64 cells. The operation also
bounds codimension-one cover candidates, coordinate integer size, and the
serialized result at 8 MiB.

The current `CubicalComplex` carrier is nonempty, so this operation does not
represent the void complex. A zero-dimensional cube is supported: its poset
has one element and no strict order relations. An all-point family produces an
antichain, also with no strict order relations between distinct cells.
