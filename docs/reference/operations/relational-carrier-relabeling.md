# Relational carrier relabeling

The `relational_structure.relabel_carrier.compute` operation applies an
explicit bijection to the finite carrier of a relational structure. Its map is
listed by source label: `old_to_new[i]` is the new label of old element `i`.
Every coordinate in every relation tuple is transported through that map;
the ranked relation signature, carrier cardinality, nullary truth values, and
complete tuple-set semantics are preserved. The result includes the inverse
map and the relabeled structure.

`csp.instance.relabel_template_carrier.compute` applies the same transport to
the target structure of a finite CSP instance. Variable labels, relation IDs,
constraint IDs, occurrence order, and ordered scopes remain unchanged. A
solution assignment is transported coordinatewise through `old_to_new`; this
gives a bijection between the source and target solution sets.

Both operations admit the complete relation tuple count and sorting work
before constructing transported tables. The CSP operation also bounds
constraint scope entries. The operation does not enumerate isomorphisms or
solutions.

These contracts use the standard finite relational structure and
homomorphism definitions: relations are transported coordinatewise, and a CSP
solution is a homomorphism into its fixed template. See the definitions in
[Barto and Opršal, “The Sherali–Adams and Weisfeiler–Leman Hierarchies in
(Promise Valued) Constraint Satisfaction Problems”](https://doi.org/10.1145/3756323).
