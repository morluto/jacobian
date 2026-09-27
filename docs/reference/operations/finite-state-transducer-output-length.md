# Subsequential transducer output length

`transducer.subsequential.output_length.compute` evaluates the length of the
output word for one input word of a partial subsequential transducer. It returns
the exact integer length without constructing the output symbols. When the
input run is undefined, it preserves the first missing transition or nonfinal
terminal state and reports the number of transition-output symbols emitted
before the function became undefined.

For an input of length `n`, each transition and final output has length at most
512, so the complete result is at most `512 * (n + 1)`. Admission bounds the
input scan and transition index by the 512-symbol word and 4,096-transition
carrier limits. This lets a caller query output lengths whose output word would
exceed the ordinary materialized-run result bound.

The operation follows the standard subsequential transducer semantics: outputs
from transitions are concatenated in input order, and a final output is appended
only when the run ends in a final-output state. See Berstel, [*Transductions and
Context-Free Languages*, Chapter 2](https://www-igm.univ-mlv.fr/~berstel/LivreTransductions/LivreTransductions.pdf).
