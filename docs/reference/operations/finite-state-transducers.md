# Finite-state transducer operations

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`transducer.subsequential.run.compute` executes one bounded partial function
realized by a deterministic subsequential transducer. A successful result
contains the initial state and state after each input symbol, one output word
per consumed symbol, cumulative transition output after every input prefix,
the separate final output, and their complete concatenation.

An undefined transition reports its first input position, state, and symbol;
the trace and partial output stop before that symbol. Ending in a state without
a final output reports `NONFINAL_DOMAIN_STATE` at input length. Successful
empty output remains `OUTPUT` with an empty result word, distinct from both
failure cases. Final output is not included in cumulative prefix outputs.

The input word is currently a tuple of integer symbol indices interpreted in
the input alphabet order carried by that run's transducer. It is explicitly
request-scoped index data, not a reusable alphabet-parented `FiniteWord`.
`FiniteWord` currently carries a structural string alphabet but no stable parent
identity, while the transducer carries an optional alphabet identity and uses
integer symbol indices. A shared parented word representation requires a
word-owner change that reconciles those two conventions; this operation does
not claim to provide that interoperability yet.

States, transitions, input length, output length, repeated prefix trace cells,
and the canonical result byte envelope are admitted before trace expansion.
The result validator checks trace shape; use
`transducer.subsequential.run.compute`'s native verifier to replay all claimed
trace fields against the source transducer and input.

`transducer.relation.inverse.compute` reverses a finite rational relation by
swapping each edge's input and output words, the corresponding alphabet sizes
and identities, and their structural alphabet values. It preserves states,
initial and accepting sets, and edge order. This is a typed relation transform;
applying it twice recovers the original canonical relation. Its work and result
size are linear in the already bounded relation representation.

`transducer.relation.projection.compute` consumes a `RationalTransducer` and
selects either its `input` or `output` tape. It returns an epsilon-NFA accepting
exactly the selected words contributed by accepting relation paths. The two
alphabets are carried separately by the relation, and the result preserves the
selected side's alphabet size, explicit ordered symbols, and optional identity.
Multiple paths and multiple outputs for one input remain relational choices;
the operation does not turn a nondeterministic relation into a function. A
zero-length selected edge label becomes epsilon, while a longer label becomes
a path with fresh intermediate states. Multiple initial states are joined by
epsilon edges from one fresh NFA start state. State, transition, work, and
result-byte estimates are admitted before expanded NFA transitions are built.
The construction is checked against an independent accepting-path enumerator
on finite acyclic relations and against inverse-relation projection identities.

`transducer.relation.outputs_for_input_automaton.compute` fixes one input word
and returns an epsilon-NFA for every output on every accepting relation path
whose concatenated input labels equal that word. The construction pairs each
transducer state with a position in the fixed word, and follows an edge only
where its entire input label matches at that position. Empty input labels do
not advance the position; multi-symbol output labels expand to NFA paths. An
output-producing input-epsilon cycle remains a cycle in the NFA, so an infinite
fiber such as `1*` is represented exactly without enumerating outputs or
determinizing the result. Product states, label matching, work, intermediate
allocation, transitions, and serialized NFA size are admitted before product
exploration. An explicit output alphabet is required so the result carries the
same reusable alphabet parent expected by regular-language operations. The
finite cases are checked against direct accepting-path
enumeration, and an epsilon-output-loop fixture checks an infinite fiber.

`transducer.relation.restrict_input.compute` intersects the relation's input
tape with a total DFA and returns the restricted rational relation. A product
state records the source relation state and DFA state. Each complete input edge
label advances the DFA before the target product state is selected; an empty
input label leaves the DFA state unchanged. Output labels, parallel edges, and
accepting-path alternatives are preserved. The result includes the exact
product-state pairs and source edge index for every result edge. The DFA must
carry the same explicit ordered alphabet and optional identity as the
relation's input side. Product exploration has an admitted work and memory
envelope, and the reachable output product must fit the rational-transducer
state and edge limits. A finite acyclic path enumerator checks the relation
identity on bounded fixtures.

This construction is the fixed-word section of a rational relation: fixing the
input to a singleton regular language and retaining the output tape produces a
regular language. It follows the classical finite-transducer and rational
transduction framework; see Berstel, [*Transductions and Context-Free
Languages*, Chapter 4](https://www-igm.univ-mlv.fr/~berstel/LivreTransductions/LivreTransductions.html).

`transducer.subsequential.from_word_morphism.compute` converts a bounded
`WordMorphism` into a one-state total transducer. The source and target alphabet
orders are preserved as explicit alphabet contexts. Every source symbol has a
self-loop whose output is its exact target-index image, and the sole final
output is empty. Empty images therefore remain defined transitions with empty
outputs. The current target carrier admits at most 32 symbols in either
alphabet and at most 512 symbols in each image; larger source morphisms remain
valid word values but are outside this transducer representation. Aggregate
mapped cells and a conservative canonical result byte estimate are checked
before transition output rows are constructed.

`transducer.subsequential.identity.compute` constructs the one-state identity
function on one explicitly ordered `FiniteAlphabet`. Every symbol has a
self-loop that emits the same index, and the empty final output fixes the
identity behavior on the empty word. Both transducer sides retain the exact
alphabet context and optional alphabet ID. The operation admits at most 32
symbols and estimates canonical result bytes before allocating transition rows.

`transducer.subsequential.reachable_states.compute` returns one shortest path
from the initial state to every reachable state, ordered by state index. When
several shortest paths exist, it chooses the lexicographically least input
word. Each row retains that input word, the complete state trace, and the
concatenation of transition outputs along the path. Final outputs are excluded:
the witness ends at a prefix state and does not assert that the state is in the
function domain. The operation bounds the worst-case witness output and
canonical result bytes before constructing path rows. Unreachable states have
no row. The result retains its source transducer so the witness paths keep
their alphabet and machine context.
