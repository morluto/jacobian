# Regular-language operations

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`regular_language.dfa.equivalence.decide` compares two total deterministic
finite automata over the same ordered integer alphabet. It explores only the
reachable product of their initial states and returns `equivalent=true` when
every reachable pair agrees on acceptance.

For inequivalent DFAs the result retains both source DFAs, the shortest
distinguishing word, and one state trace per source automaton. Product
successors are expanded in ascending alphabet order, so the witness is the
lexicographically least word among all shortest witnesses. The empty word is
considered first. A zero-symbol alphabet is valid and has only the empty word;
DFAs with a positive alphabet must still provide exactly one transition for
every state-symbol pair.

Product-state, transition, predecessor, witness, work, and result-allocation
envelopes are admitted before traversal. Resource refusal is not an equivalence
result. The result model checks witness and trace shape without replaying the
automata; use `regular_language.run.check` when an independent run result is
needed.
