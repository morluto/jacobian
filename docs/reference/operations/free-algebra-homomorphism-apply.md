# Free associative algebra homomorphism application

`free_algebra.homomorphism.apply.compute` evaluates the unique unital
`QQ`-algebra homomorphism specified by a source alphabet, a target alphabet,
and one target polynomial image for each source generator. It substitutes each
word in its original letter order, extends linearly, and collects like target
words into the canonical descending degree-lexicographic representation.

The request preserves both axes, including empty alphabets. An empty source
word maps to the target unit; a zero generator image annihilates every word
containing that generator. The operation is exact and noncommutative: image
factors are multiplied in source-word order.

The current execution envelope admits at most 26 letters per alphabet, 64
source terms, 64 terms in each generator image, 32 letters per input/image
word, 4096 distributive contributions, 64 letters per output word, and
64 decimal digits per predicted rational coefficient component. Admission
checks the expansion, word length, and coefficient growth before expansion.

For example, with `x ↦ u+v` and `y ↦ uv`, the commutator `xy-yx` maps to
`uuv + vuv - uvu - uvv`; the order of the two products is retained.
