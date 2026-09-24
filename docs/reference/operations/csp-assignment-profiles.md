# Finite CSP assignment profiles

[Operation references](index.md) · [Tool surface](../tools.md)

`csp.assignment.profile.compute` evaluates one total map from the instance's
variable axis to its template carrier. It returns `SOLUTION` exactly when every
named constraint occurrence evaluates to a tuple in the named template
relation. Otherwise it returns `NOT_A_SOLUTION` and retains the first failed
occurrence in the instance's declared order. The result also lists the exact
assigned tuple and membership outcome for every occurrence, including repeated
or duplicate constraints.

The operation checks a supplied assignment only. A failing assignment says
nothing about whether another assignment solves the instance. Unsatisfiability
requires complete bounded search or another operation with an exact
unsatisfiability contract.

Before evaluation, the instance is admitted against the finite CSP bounds of
4,096 constraint occurrences and 16,384 aggregate scope entries. The complete
assignment has exactly one in-range template value for each variable. Empty
variable axes and empty constraint families are valid: the unique empty map is
a solution when there are no constraints.

The instance can be converted separately with
`csp.instance.to_source_structure.compute`. That canonical source structure
maps homomorphically to the template exactly when an assignment is a solution;
the assignment profile retains occurrence IDs that the ordinary relation
table intentionally deduplicates.
