# Regular tree grammars and bottom-up automata

`regular_tree_grammar.to_automaton.compute` converts a finite unit-free
regular tree grammar into the existing nondeterministic bottom-up tree
automaton value. The convention is the standard ranked production
`A -> f(B1,...,Bk)`: each nonterminal is an automaton state, each production
becomes the transition `f(B1,...,Bk) -> A`, and the grammar's start
nonterminal is the sole final state. This is the regular-tree-grammar to
tree-automaton correspondence described in [Tree Automata Techniques and
Applications, Chapter 2](https://jacquema.gitlabpages.inria.fr/files/tata.pdf).

`tree_automaton.to_regular_tree_grammar.compute` constructs a grammar whose
ground-tree language is exactly the accepted language of a bottom-up
automaton. A single final state serves as the grammar start. With multiple
final states, the converter adds one synthetic start nonterminal and copies
each distinct ranked transition with a final target into a production from
that start; it keeps the original transition productions so those states can
still derive child subtrees. If there are no final states or no nullary
transition seed, it returns an empty grammar over the same ranked signature.
The generated start rules implement the union of the final-state languages
without unit productions.

Nonterminals and symbols use explicit integer IDs. The arity tuple is the full
ranked signature, including symbols not used by a production. Input production
order is normalized to a canonical order; duplicate rules are rejected. The grammar retains unreachable
and nonproductive nonterminals, and conversion preserves those states rather
than minimizing the automaton. A grammar with no productions accepts the empty
tree language, including the empty-signature case.

The versioned value omits unit productions `A -> B`, epsilon rules, semantic
actions, and unranked symbols. Its productions are limited by the tree
automaton's 64-state, 32-symbol, 16-arity, and 4,096-transition bounds. The
conversion admits work from production count, comparison-key ranks, and
child-axis lengths before
creating the automaton transitions. Its output has exactly one transition per
production and retains the bounded source grammar, so the same domain limits
bound the complete result size. The result composes directly with tree runs and
counting. Boolean products require complete deterministic automata, while
minimization requires a deterministic automaton; arbitrary grammar conversions
may need determinization (and completion for Boolean products), which can change
state IDs.

The reverse conversion bounds the exact copied root-production count, grammar
nonterminals, production ordering work, and serialized source-plus-grammar
cells before constructing productions. A multiple-final automaton at the
maximum 64 states may exceed the grammar carrier because it needs a synthetic
65th start nonterminal; that request receives a typed resource refusal. Copying
multiple final-target transitions can also exceed the 4,096-production bound.
