# Antidominant Weyl representatives

`weyl_group.antidominant_representative.compute` returns the unique
antidominant weight in the finite Weyl orbit of an integral weight, together
with an exact Weyl element carrying the source weight to that representative.
Coordinates are in the fundamental-weight basis and remain bound to the input
Cartan matrix. An input already in the closed antidominant chamber, including
zero, returns the identity transporter.

For other inputs, the operation moves the weight to the dominant chamber and
then applies the longest Weyl element `w0`. Since `w0` maps the dominant chamber
to the antidominant chamber, the resulting transporter is the composition
`w0 w`; the returned `root_action` is the exact action on the simple-root
basis. The kernel checks that its longest-element word has length equal to the
complete positive-root count before constructing the result.

Admission bounds the finite Cartan datum, source and orbit coordinates,
chamber-normalization reflections, longest-word construction and inversion
verification, action-matrix updates, and serialized result size before those
expansions. The finite-type chamber and longest-element conventions agree with
the [Sage root-lattice realization reference](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/root_system/root_lattice_realizations.html).

For `A2` and source weight `(-1, 1)`, the unique antidominant orbit weight is
`(0, -1)`.

[Operation references](../index.md) · [Tool surface](../../tools.md)
