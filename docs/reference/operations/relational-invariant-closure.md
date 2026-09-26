# Finite polymorphism-invariant relation closure

`relation.closure_under_operations.compute` returns the least relation on
the selected power `A^r` that contains the supplied generator tuples and is
closed under coordinatewise application of each supplied operation. The
operation tables are checked against every basic relation of the exact source
structure before they are used. A bad operation is rejected with its first
preservation failure.

The result retains the source, relation arity, generators, and operations,
along with the canonical sorted closure tuples. It is an ordinary finite
relation: every row is in `A^r`, generators are included, and closure is exact.
For `r = 0`, the ambient power contains only the empty tuple. Positive-arity
operations on the empty carrier have empty tables and act on the unique nullary
row in the usual way.

Admission bounds the supplied operation tables, the preservation work, and the
fixed result frames before any product of tuples is expanded, then charges the
tuple, work, and output envelopes against the reachable closure as the kernel
generates it. The reachable closure can be far smaller than the ambient power
`A^r`, so a small seed set is never refused for the surrounding power.
Exceeding the envelope is a resource refusal, never an incomplete relation
result. The operations are caller-supplied generators; this operation does not
enumerate or claim a complete polymorphism clone.

The mathematical postcondition is the finite generated-subalgebra closure in
the direct power `A^r`, the same coordinatewise construction used to derive
relations invariant under polymorphism operations. See the exact finite
preservation and closure definitions in [CSP and finite relational
structures](https://drops.dagstuhl.de/storage/02dagstuhl-follow-ups/dfu-vol007/DFU.Vol7.15301/DFU.Vol7.15301.pdf).
