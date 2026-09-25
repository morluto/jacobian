# Accepted ranked trees by height

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`tree_automaton.accepted_tree_height_profile.compute` returns the exact number
of distinct accepted ranked trees of height at most `h`, for every `h` from
zero through `max_height`. Leaves have height zero. Its complete deterministic
input gives every tree exactly one root state, and the recurrence extends the
count vector through each ranked transition. This is a height prefix; the
separate `tree_automaton.accepted_tree_count.compute` operation counts trees of
one exact node size and also supports nondeterministic automata.

The operation admits transition work, a conservative all-tree integer digit
bound, and aggregate output bytes before recurrence evaluation. A large
all-tree bound may refuse an automaton whose accepted language is smaller; an
empty final-state set is handled directly and returns a zero profile. Height
is bounded at 100, and resource refusal does not imply a count or language
property.
