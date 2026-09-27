# Restricting a rational transducer's output language

`transducer.relation.restrict_output.compute` intersects a finite rational
relation `R ⊆ A* × B*` with the regular output language recognized by a total
DFA `L ⊆ B*`. Its result recognizes exactly `R ∩ (A* × L)`.

The operation constructs the reachable product of transducer states and DFA
states. It advances the DFA through each complete output edge label, preserves
input and output edge labels and nondeterministic alternatives, and returns
transports from each product state and output edge to its source data. The
request supplies the `FiniteAlphabet` that binds the DFA's integer symbol axis
to the transducer output axis. If the transducer already carries that context,
the contexts must agree exactly.

The operation admits the full product work bound before traversal, limits the
reachable product to 64 states and 4,096 edges, and bounds aggregate copied
label cells. Exceeding a limit returns an admission error, never a partial
restricted relation.
