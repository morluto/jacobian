# Boolean products of deterministic tree automata

`tree_automaton.boolean_product.compute` constructs the synchronous product of
two complete deterministic bottom-up tree automata over the same ordered
ranked signature. The `connective` selects intersection, union, left
difference, or symmetric difference. A product state is an ordered pair of
source states, enumerated lexicographically, and `state_pairs` preserves that
state-axis correspondence.

For each symbol, product transitions pair the component transition rows in
child-position order. Completeness and determinism give exactly one successor
for every product-state tuple. The output final-state predicate is respectively
`left and right`, `left or right`, `left and not right`, or exclusive-or.
Partial and nondeterministic inputs are rejected: in particular, a raw
synchronous product of partial machines would not implement union when one
component gets stuck. Use `tree_automaton.deterministic.complete.compute` to
add a nonfinal sink state to a partial deterministic automaton before taking
the product.

The operation checks equal signatures, deterministic complete transition
tables, the Cartesian product state count, and the exact paired transition-row
count and work before allocating the product. The output is a complete
deterministic automaton and can be passed directly to run, count, or complement
operations.

## Completing a partial deterministic automaton

`tree_automaton.deterministic.complete.compute` fills every missing transition
with one appended, nonfinal sink state. Existing state IDs, final states, and
the ranked alphabet are preserved. If the input is already complete, the
operation returns the same state space without adding a sink. A missing
transition makes a partial run reject; routing that case to a nonfinal sink
preserves the accepted ground-tree language. State, row, output-cell, and work
bounds are checked before enumerating child-state tuples.
