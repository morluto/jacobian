# Exact rational linear systems

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`linear.rational_solution.compute` returns one exact bounded rational solution
candidate for a supplied system. `linear.rational_inconsistency.compute` returns
one normalized left witness when the supplied system is inconsistent.

Both operations take the entire typed system in their request and return their
candidate or witness inline. They do not consume producer IDs, create evidence
records, or require a checker-installation decision.

The system stores its coefficient matrix canonically as row-major nonzero
coordinates. `row_count`, the ordered variable axis, and the right-hand side
retain implicit zero rows and columns without materializing a dense matrix.
Duplicate coordinates, stored zeros, and coordinates outside those axes are
rejected. Admission separately bounds axes, stored nonzeros, rational digits,
the conservative dense fill-in work envelope, and the exact output-height
estimate before SymPy's maintained sparse `DomainMatrix` RREF kernel runs.
