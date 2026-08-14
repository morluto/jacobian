# SAT and SMT operation references

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The logic tools take and return bounded inline values.

- `sat.cnf.canonicalize` turns named clauses into a canonical CNF value.
- `sat.assignment.check` checks one complete Boolean assignment against that
  value.
- `sat.solve` solves one canonical CNF with the maintained Z3 Python binding.
- `smt.solve` solves one bounded QF_UF, QF_LIA, or QF_LRA SMT-LIB query with
  the same binding.

The solver result is `SAT`, `UNSAT`, or `UNKNOWN`. A SAT result contains a
bounded model; `UNKNOWN` makes no mathematical conclusion. There are no CNF,
model, proof, or solver-result URIs.
