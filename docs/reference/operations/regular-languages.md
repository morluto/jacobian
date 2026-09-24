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

`regular_language.subsequential_transducer.preimage.compute` returns a total
DFA for inputs on which a deterministic subsequential transducer is defined
and whose emitted word belongs to the target DFA language. Each transition
output is consumed in order, followed by the terminal state's final output.
An undefined transition or a terminal state without a final output rejects;
the returned DFA includes a rejecting sink when needed. The target DFA and
transducer output must carry the same explicit ordered `FiniteAlphabet` and
matching optional identity. The result retains the transducer's explicit input
alphabet parent and identity. Product exploration work, reachable DFA size,
transition count, and a conservative result-byte envelope are bounded before
the search.

`transducer.subsequential.regular_image.compute` returns an `NFA` recognizing
the regular output language obtained by applying a partial subsequential
transducer to words of a source DFA. It first forms reachable
source-state/transducer-state pairs. Each defined input transition contributes
its output word as a path over the output alphabet with fresh intermediate
states; an empty transition output becomes an explicit epsilon edge. A pair is
accepting only when the source state accepts and the transducer state has a
final output, which is appended as another output path to a shared accepting
state. Undefined transitions add no path. The result carries the transducer's
explicit output alphabet and identity.

The request requires the DFA alphabet and transducer input alphabet to have the
same explicit ordered symbols and optional identity. Admission bounds the full
product, expanded epsilon-NFA states and edges, construction work, intermediate
allocation, and serialized output before expansion. NFA transitions have
contiguous IDs and use a null symbol only for epsilon; parallel paths remain
distinct. An NFA's alphabet size, optional identity, and explicit ordered
alphabet context define the meaning of every non-epsilon symbol index. The
output-path expansion is the standard finite-automaton construction; see the
[NFA construction and subset-construction discussion in these automata
notes](https://www.cis.upenn.edu/~jean/old511/html/cis51108sl2.pdf#page=38).

`regular_language.nfa.membership.decide` checks one finite word in an explicitly
parented NFA. It computes epsilon closure before reading and after each symbol,
propagates the reachable state set along matching labeled edges, and accepts
exactly when the final state set meets the NFA's accepting states. The word is
a sequence of zero-based symbols on the NFA's own ordered alphabet. State,
edge, epsilon-closure, word-length, and total work bounds are checked before
propagation.
