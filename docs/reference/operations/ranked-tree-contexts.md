# One-hole ranked-tree contexts

`ranked_tree.context.plug.compute` substitutes a ranked ground tree into the
unique hole of a finite tree context. A context is represented canonically by
its exact ranked alphabet and the root-to-hole spine. Each spine frame stores
the parent symbol, the child position containing the hole, and every other
child tree in order. The empty spine is the identity context.

This is the standard one-hole context convention: a context is a tree with one
distinguished occurrence, and `C[t]` replaces that occurrence with `t`. See
[Tree Automata and Applications, section 1.4](https://lsv.ens-paris-saclay.fr/~schwoon/enseignement/tata/tata.pdf)
for the definitions of contexts and substitution.

Context siblings and the plugged tree must match the exact stored ranked
alphabet. The operation admits total output node count and depth before it
constructs any result tree. The current envelope is 4,096 nodes and depth 128.

`tree_automaton.context.state_map.compute` returns the total function
`q -> state(C[q])` for a complete deterministic bottom-up automaton. It
propagates each state up the context spine; ground siblings are evaluated once.
The automaton and context must have identical ranked alphabets. This operation
provides the reusable context action needed by later contextual-equivalence
and tree-language algebra work. It does not compute an induced transformation
monoid or distinguishability witnesses.
