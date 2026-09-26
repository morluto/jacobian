# Petri-net disjoint union

`petri_net.disjoint_union.compute` forms the disjoint union of two weighted
place/transition nets. It concatenates the source place and transition axes,
placing the left axes first, then the right axes. The input and output arc
matrices are block diagonal:

\[
\mathrm{Pre}_{L\sqcup R}=\begin{pmatrix}\mathrm{Pre}_L&0\\0&\mathrm{Pre}_R\end{pmatrix},\qquad
\mathrm{Post}_{L\sqcup R}=\begin{pmatrix}\mathrm{Post}_L&0\\0&\mathrm{Post}_R\end{pmatrix}.
\]

The result includes both source nets and the canonical injections of their
place and transition axes. When both source markings are supplied, their token
vectors are concatenated and bound to the union net. Supplying only one source
marking is invalid. Existing source IDs are namespaced by side and position in
the union, so equal labels from separate components remain distinct; when both
source axes are unlabeled, the union axis remains unlabeled.

The firing equation \(M' = M - \mathrm{Pre}_{t} + \mathrm{Post}_{t}\) acts
componentwise on a block-diagonal net. A transition from one component leaves
the other marking unchanged. For bounded source reachability graphs, the union
reachability graph has the Cartesian product of the source state sets and the
componentwise transition edges. This operation constructs the net and optional
initial marking; it does not construct that reachability graph.

Admission checks the combined place and transition dimensions, matrix-entry
work, and a conservative serialized-output bound before constructing the
block matrices and result. The supported carrier limit is 64 places and 64
transitions.
