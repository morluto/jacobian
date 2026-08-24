# SAT and SMT operation references

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The logic tools take and return bounded inline values.

- `sat.cnf.canonicalize` turns named clauses into a canonical CNF value.
- `sat.assignment.check` checks one complete Boolean assignment against that
  value.
- `sat.solve` solves one canonical CNF with the maintained Z3 Python binding.
- `sat.refutation.check` replays one typed `LPR_ASCII_V1` refutation against
  that exact canonical CNF. The proof carries typed additions, deletions,
  asymmetric-tautology hints, and propagation hints—not raw checker bytes,
  flags, paths, or logs. The CakeML `cake_lpr` backend is installed only in
  the pinned Linux/amd64 service image; other installations return
  `UNAVAILABLE`.
- `smt.solve` solves one bounded QF_UF, QF_LIA, or QF_LRA SMT-LIB query with
  the same binding.

The solver result is `SAT`, `UNSAT`, or `UNKNOWN`. A SAT result contains a
bounded model; `UNKNOWN` makes no mathematical conclusion. A refutation result
is source-bound: only `VALID_REFUTATION` establishes UNSAT.
`INVALID_REFUTATION` says the supplied typed derivation does not establish
contradiction; it never establishes SAT. Timeout, unavailable backend, and
execution errors are non-conclusions. There are no CNF, model, proof, or
solver-result URIs.
