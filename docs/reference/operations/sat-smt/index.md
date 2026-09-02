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

## Solver budgets

`smt.solve` admission bounds the source before Z3 parses it: ASCII bytes,
nesting depth, compound terms, declared symbols, and decimal numeral width.
Each budget names the quantity that controls parser work, solver
preprocessing, symbol-table size, or big-integer expansion. Both solvers give
Z3 a complete request-scoped budget: wall-clock time, a deterministic work
limit (`rlimit`), and a memory ceiling (`max_memory`). Exhausting any of them,
or exceeding the bounded model size, returns `UNKNOWN` with an `exhausted`
value of `time`, `work`, or `memory` where applicable; it never raises a host
exception or yields a mathematical conclusion.
