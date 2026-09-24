# Regular tree grammars and bottom-up automata

`regular_tree_grammar.to_automaton.compute` converts a finite unit-free
regular tree grammar into the existing nondeterministic bottom-up tree
automaton value. The convention is the standard ranked production
`A -> f(B1,...,Bk)`: each nonterminal is an automaton state, each production
becomes the transition `f(B1,...,Bk) -> A`, and the grammar's start
nonterminal is the sole final state. This is the regular-tree-grammar to
tree-automaton correspondence described in [Tree Automata Techniques and
Applications, Chapter 2](https://jacquema.gitlabpages.inria.fr/files/tata.pdf).

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
bound the complete result size. The result composes directly with tree runs,
counting, Boolean operations, and minimization without changing the signature
or state IDs.
