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

States, transitions, input length, output length, and the aggregate
cumulative/transition output cells admitted by the trace allocation bound are
checked before trace expansion.
The result validator checks trace shape; use
`transducer.subsequential.run.compute`'s native verifier to replay all claimed
trace fields against the source transducer and input.

`transducer.relation.inverse.compute` reverses a finite rational relation by
swapping each edge's input and output words, the corresponding alphabet sizes
and identities, and their structural alphabet values. It preserves states,
initial and accepting sets, and edge order. This is a typed relation transform;
applying it twice recovers the original canonical relation. Its work and result
size are linear in the already bounded relation representation.

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
symbols; its transition, final-output, and symbol-string allocations are fixed
by the admitted alphabet cardinalities and the validated symbol-length bound.

`transducer.subsequential.reachable_states.compute` returns one shortest path
from the initial state to every reachable state, ordered by state index. When
several shortest paths exist, it chooses the lexicographically least input
word. Each row retains that input word, the complete state trace, and the
concatenation of transition outputs along the path. Final outputs are excluded:
the witness ends at a prefix state and does not assert that the state is in the
function domain. The operation bounds the aggregate worst-case witness output
cells before constructing path rows. Unreachable states have
no row. The result retains its source transducer so the witness paths keep
their alphabet and machine context.
