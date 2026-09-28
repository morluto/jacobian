# Tree context transformation monoid

`tree_automaton.context.transformation_monoid.compute` takes a complete
deterministic bottom-up tree automaton and returns every state map induced by
one-hole ranked-tree contexts. The result includes the finite multiplication
table, the identity element, and one explicit ranked context for each map.

The map of a context `C` sends a state `q` to the state obtained by evaluating
`C` with `q` at its hole. Multiplication is function composition: table entry
`(C, D)` is the map `q ↦ C(D(q))`. The empty context is the identity. Maps are
stored once in lexicographic order on their state images.

The operation generates elementary contexts with one hole child and fills
other children with canonical minimum trees for reachable states. These
generate all contexts because every sibling of a one-hole context is a ground
tree. A witness context is retained for each distinct map so the result
composes with context plugging and state-map evaluation.

`max_elements` bounds the exact monoid size. The kernel first admits the
mandatory reachable-state profile (transition sorting, bounded saturation
scans, and witness materialization) from the same fixed work envelope, then
admits elementary generator work, closure work, exact multiplication-table
work and cells, and the aggregate size and depth of witness contexts. If any
bound is exceeded, the operation refuses the request without returning a
partial monoid.

The context and transformation definitions follow [Tree Automata Techniques
and Applications, section 1](https://jacquema.gitlabpages.inria.fr/files/tata.pdf).
