# Exact rational polytope joins and prisms

[Geometry operations](index.md) · [Tool surface](../../tools.md)

\`polytope.rational.prism.compute\` and \`polytope.rational.join.compute\`
construct exact rational V-polytopes from bounded, nonempty rational
V-presentations. Their results retain the vertical coordinate name explicitly
as \`height_axis\`, as well as source-to-result vertex maps.

For a polytope \(P\) on ordered axes \(A\), the prism operation returns
\(P \times [0,1]\) on \`(*A, height_axis)\`. Each source vertex \`p\` produces
\`(p, 0)\` and \`(p, 1)\`. The bottom and top maps retain the source and result
vertex IDs, and identify the side. Its affine dimension is
\`dim(P) + 1\`.

For factors \(P\) and \(Q\) on disjoint ordered axes \(A\) and \(B\), the join
operation embeds each vertex of \(P\) as \`(p, 0, 0)\` and each vertex of \(Q\)
as \`(0, q, 1)\` on \`(*A, *B, height_axis)\`. The left and right maps retain
their source and result IDs and identify the factor. Its affine dimension is
\`dim(P) + dim(Q) + 1\`.

The height axis must be fresh. Join factors must also have disjoint axis and
vertex labels. The current input type admits at most seven axes; each result
is rejected before construction if its vertex rows exceed the polytope
envelope. Exact coordinate transport introduces only zero and one, so it
does not increase coordinate height. The results and maps are ordinary typed
values that can be serialized and consumed directly.
