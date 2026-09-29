# Generalized arc consistency for finite CSPs

[Documentation home](../../index.md) · [Tool surface](../tools.md) · [Operation references](index.md)

`relational.csp.generalized_arc_consistency.compute` prunes the domain of every
variable until each surviving value is supported by every constraint in which
that variable occurs. It is a fixpoint computation, not a search: it reports
which values cannot participate in any solution, not whether a solution exists.

## Completeness

The operation reasons only over the supplied instance and the supplied initial
domains. It does not decide satisfiability. A domain that survives pruning is
consistent with every single constraint taken alone, which is weaker than
global consistency: a surviving combination can still be unsatisfiable.

## Exactness

Every domain is a set of exact carrier labels, and the result reports the
retained labels per variable in variable order. Candidate rows are
occurrence-local after exact duplicate relation-and-scope profiles are
coalesced, so a repeated constraint costs one profile rather than one pass per
occurrence.

## Relation to the other CSP operations

`csp.assignment.profile.compute` evaluates one complete assignment against
every named constraint, and `csp.solutions.enumerate.compute` enumerates
solutions outright. Generalized arc consistency sits between them: it removes
values that no solution can use, which is much cheaper than enumeration and
much weaker than a decision procedure. A consumer that needs a satisfiability
answer must still call an enumerating or decision operation.
