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

`regular_language.symbol_parikh_profile.compute` returns the complete exact
histogram of accepted words of one length, keyed by dense vectors of symbol
multiplicities on the DFA's ordered alphabet axis. Paths are merged as soon as
their state and symbol counts agree, so transitions carrying the same symbol do
not create a finer transition-level profile. Cells are lexicographically
ordered and omit zero multiplicities; the source DFA, alphabet axis, length,
and aggregate accepted-word count remain attached for composition.

The operation admits transition-index construction, reachable-state discovery,
all extended weak-composition DP layers, vector-coordinate updates, final-layer
scans, exact count digits, and complete output materialization before running.
A zero-symbol DFA admits only the empty word; a positive alphabet is still
required to be a complete transition carrier.
