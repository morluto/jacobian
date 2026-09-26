# Complementing a deterministic bottom-up tree automaton

`tree_automaton.complement.compute` complements a complete deterministic
bottom-up tree automaton over its existing finite ranked signature. The
`arity` tuple is the ordered signature: symbol IDs are its indices, and each
symbol's arity is the corresponding entry.

For state count `q`, completeness requires one transition for every pair
`(symbol, child_state_tuple)`, with `q ** arity[symbol]` tuples for each
symbol. Duplicate transition keys are nondeterministic and rejected. Missing
keys are incomplete and rejected. Before constructing output rows, the
operation bounds the exact transition product by 4,096 and checks the
result-cell envelope.

The output preserves the signature and state count, remaps states by first
occurrence in the lexicographically ordered transition table, and appends
states absent from that table in source order. It returns both directions of
the bijective state map. The output transition table is sorted canonically.
The complement is obtained by replacing the final set `F` by its complement
in the complete state set, transported through the state map. Because the
input is deterministic and complete, each finite ground tree has exactly one
run, so this final-state exchange accepts precisely the trees rejected by the
input.

An empty ranked signature is allowed. It has no ground trees and an empty
transition table; the complement operation is still a well-defined finite
automaton transform over that same signature.
