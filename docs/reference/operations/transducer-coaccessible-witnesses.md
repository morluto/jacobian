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
that every witness fits the 4,096-symbol result limit; with at most 64 rows,
this bounds the complete witness output to 262,144 symbols. It returns no
partial witness set on resource refusal.
