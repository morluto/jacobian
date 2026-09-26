# Restrict a subsequential transducer to a regular domain

`transducer.subsequential.restrict_domain.compute` accepts a canonical partial
subsequential transducer `T : A* -> B*` and a total DFA `D` over the exact same
ordered alphabet `A`. It returns the subsequential transducer for the partial
function

```text
x |-> T(x), when x is in dom(T) and D accepts x.
```

The DFA alphabet size, optional identity, and optional `FiniteAlphabet` context
must match the transducer input alphabet. Equal alphabet sizes alone do not
make differently parented symbol indices interchangeable.

The kernel explores the reachable product of transducer and DFA states. A
product state is final exactly when the transducer state has a final output and
the DFA state is accepting. It removes product states that cannot reach a final
product state and omits transitions into those states: such a transition cannot
belong to any accepted input, so omitting it preserves the restricted partial
function. If the initial product state is not coaccessible, the result is the
one-state transducer with no transitions or final output, representing the
empty function while retaining both alphabets.

The product has at most `64 * 64` states and `64 * 64 * 32` candidate
transitions under the source carrier limits. The operation preflights the
product scan, reverse coaccessibility traversal, source validation, and bounded
output-label validation/copying. The returned value must fit the canonical
transducer carrier: at most 64 states, 4,096 transitions, and 512 output
symbols per transition or final output. Requests whose trimmed product exceeds
those carrier limits return a typed resource refusal without a partial
transducer.

This is the standard synchronous product construction for intersecting a
regular domain with a finite-state transduction. It composes directly with the
existing finite-word, DFA, and subsequential-transducer values; it does not
change either alphabet parent or any output word.
