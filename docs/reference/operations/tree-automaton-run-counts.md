# Counting nondeterministic tree-automaton runs

`tree_automaton.nondeterministic.run_counts.compute` returns the exact number
of accepting runs for each tree size from one through `max_size`. A run is a
ranked tree together with one state assignment at every node such that every
node and its ordered child-state tuple match a transition, and the root state
is final. If one tree has two accepting state assignments, it contributes two
runs. This differs from `tree_automaton.accepted_tree_count.compute`, which
counts each accepted tree once.

The result profile is indexed by size minus one: entry zero counts one-node
trees. Nullary transitions contribute at size one; sizes with no possible
trees or no final states contribute zero. The operation uses a sum/product
dynamic program over transition rows and admits its transition convolution
work and exact coefficient/profile digit bounds before computation.
