# Deterministic tree-automaton state algebra

`tree_automaton.deterministic.state_algebra.compute` returns the finite
single-sorted algebra defined by a complete deterministic bottom-up tree
automaton's transition functions. Its carrier position `i` is automaton state
`i`, including states that no ground tree reaches. Ranked symbol `j` becomes
the algebra operation `tree_symbol_j`, with the same arity as symbol `j`.
Operation-table rows follow the lexicographic order of child-state tuples.

The output is the transition algebra used to evaluate ground ranked trees as
terms. It does not include the automaton's accepting-state subset, which is a
separate part of the recognizer. Keeping every source state and the symbol
index/arity convention makes the transition functions directly usable by
finite universal-algebra operations. This is the standard interpretation of a
complete deterministic tree automaton as a finite algebra, described in
[Tree Automata Techniques and Applications](https://www.lsv.fr/~schwoon/enseignement/tata/tata-compress.pdf).

The conversion admits the existing finite-algebra envelope before constructing
tables: at most 32 carrier states, 16 operation symbols, arity 4, and 65,536
table cells. The complete deterministic input already provides every cell of
each operation table. Incomplete automata must first be completed explicitly.
