# Free word factor-avoidance automata

`free_algebra.factor_avoidance.dfa.compute` returns an exact total DFA for
words over a finite ordered free-generator alphabet that avoid a supplied
finite family of contiguous factors. DFA symbols are generator ranks in the
declared alphabet order. The returned carrier retains that ordered alphabet
and the sorted distinct source pattern family.

The construction tracks the longest suffix of the scanned word that is a
proper prefix of a forbidden factor. A sink state records that a forbidden
factor has occurred. The construction internally removes duplicate patterns
and patterns containing another forbidden pattern because they do not change
the language; the result preserves the canonical sorted distinct source
family.

The operation admits at most 32 patterns, 2,048 supplied letters, 64 DFA
states, 15,000,000 work units, and 150,000 output cells. It rejects a request
whose minimal prefix-state construction exceeds the shared regular-language
DFA carrier. An empty forbidden factor yields the empty language; an empty
family yields the full free monoid, including when the alphabet is empty.

This operation describes exactly the finite pattern-avoidance language from
its input. It does not assert that a finite-degree Gröbner–Shirshov leading-word
prefix is globally complete for an ideal, so a DFA built from such a prefix is
not by itself the quotient's global normal-word language.

The construction is the standard trie/failure-state pattern-matching method of
Aho and Corasick, specialized to the exact language that has not yet matched a
forbidden word: [Aho–Corasick, “Efficient String Matching” (1975)](https://cr.yp.to/bib/1975/aho.pdf).
