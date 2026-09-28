# Complete finite CSP solution enumeration

[Operation references](index.md) · [Tool surface](../tools.md)

`csp.solutions.enumerate.compute` returns every satisfying assignment of one
finite CSP instance. Each assignment is a tuple on the original variable axis,
with values in the template carrier. Results are unique and in lexicographic
order. The result retains the complete instance, including named constraint
occurrences that share a scope and therefore deduplicate in its canonical
source relational structure.

The operation constructs that source structure and reuses complete relational
homomorphism enumeration into the template. Its mathematical postcondition is
exactly the same as direct constraint evaluation: a returned assignment
satisfies every named occurrence, and every satisfying assignment is returned.
No solver, sampling, or truncation is used.

Before enumeration, admission bounds the complete map space, tuple-replay work,
and maximum retained assignment labels. The current limits admit at most 65,536
candidate assignments and 1,048,576 assignment labels. A request outside those
limits is rejected; it does not return a partial family. Empty variable axes
have one candidate, the empty assignment; a nonempty variable axis mapped into
an empty template has no candidates. Nullary true and false relations are
handled by the same exact relation-preservation rule.
