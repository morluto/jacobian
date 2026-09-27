# Subsequence-transducer coaccessibility witnesses

`transducer.subsequential.coaccessible_states.compute` returns one witness for
each state from which a function-domain word can finish. It selects a suffix
with minimum input length, then the lexicographically first suffix in the
ordered integer input alphabet. The witness includes its state trace, source
transition indices, terminal state, and complete output word, including the
terminal state's final output.

For a state already carrying a final output, the selected suffix is empty. A
state without any successful continuation has no row. The operation preserves
the source transducer as the alphabet and transition context. It performs a
reverse breadth-first search over at most 64 states and 4,096 transitions.
Each path has at most 63 transitions. Before building output words, it checks
that each witness fits the 262,144-symbol bound and that all witnesses together
fit the same 262,144-symbol aggregate budget. These limits are each 64 times
the 4,096-symbol transducer result-word limit. It returns no partial witness
set on resource refusal.
