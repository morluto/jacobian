# SAT and SMT operation references

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The logic tools take and return bounded inline values.

- `sat.cnf.canonicalize` turns named clauses into a canonical CNF value.
- `sat.assignment.check` checks one complete Boolean assignment against that
  value.
- `sat.solve` solves one canonical CNF with the maintained Z3 Python binding.
- `smt.solve` solves one bounded QF_UF, QF_LIA, QF_LRA, QF_NRA, or QF_BV
  SMT-LIB query with the same binding.
- `smt.unsat_core` returns SAT, UNKNOWN, or an indexed UNSAT core of its source
  assertions.

The solver result is `SAT`, `UNSAT`, or `UNKNOWN`. For SAT, `sat.solve`
returns an assignment and `smt.solve` returns a bounded model projection;
`smt.unsat_core` returns no core. `UNKNOWN` makes no mathematical conclusion.
There are no CNF, model, proof, or solver-result URIs.

`QF_NRA` carries an exact source-bound model only when every component fits the
canonical rational or real-algebraic carrier. `QF_BV` uses fixed-width modular
semantics and returns Z3's bounded SMT-LIB display projection; that projection
is for inspection, not a reusable typed bit-vector assignment. A caller needing
word-level composition must retain the original query until a future
source-bound assignment carrier is admitted.

## Solver budgets

`smt.solve` admission bounds the source before Z3 parses it: ASCII bytes,
nesting depth, compound terms, declared symbols, and decimal numeral width.
For `QF_BV`, it additionally bounds each sort to 1,024 bits and the aggregate
bit-width of distinct parsed terms to 65,536 bits. The admitted operators cover
fixed-width arithmetic, shifts, bitwise logic, concatenation, extraction, and
signed/unsigned comparisons; quantifiers, conversions, floating-point terms,
and uninterpreted functions are rejected. Thus modular overflow and signedness
remain explicit source semantics rather than an implicit conversion to integers.
Each budget names the quantity that controls parser work, solver
preprocessing, symbol-table size, or big-integer expansion. Both solvers give
Z3 a complete request-scoped budget: wall-clock time, a deterministic work
limit (`rlimit`), and a memory ceiling (`max_memory`). Known work, memory,
time, or result-output exhaustion raises an execution exception. A healthy
inconclusive solver answer remains `UNKNOWN`; the existing success schemas
are unchanged. The `exhausted` field on SAT/SMT success results is now null.

The work allowance of `sat.solve` and `smt.solve` is fixed: increasing
`timeout_ms` does not increase it. `smt.unsat_core` exposes `rlimit`; callers
may adjust it within the range advertised by its request schema. `timeout_ms`
defaults to 10,000 milliseconds and admits values from 1 through 120,000
milliseconds for all three operations. It is a full-lifecycle wall limit: all
mandatory phases share the parent deadline, including worker startup, parsing,
solving, result validation, and projection. An earlier caller or transport
deadline still wins.

Native callers receive `OperationResourceExhaustedError` for work, memory, or
output capacity, `OperationExecutionTimeoutError` for time expiry, and
`OperationBackendError` for initialization, startup, abnormal exit, malformed
response, or invalid backend mathematics. Cancellation preserves
`OperationExecutionCancelledError`. These transport-independent exceptions
have safe messages; their private causes are not mathematical output.

MCP uses the SDK's
[ToolError handling](https://py.sdk.modelcontextprotocol.io/servers/handling-errors/):
`is_error=True`, model-visible text, and no error `structured_content`. Text
contains `RESOURCE_EXHAUSTED`, `OPERATION_TIMEOUT`, `OPERATION_CANCELLED`, or
`OPERATION_FAILED`, with the operation ID, stage, safe message, and applicable
resource and recovery hint. Backend defects and unexpected exceptions receive
one private traceback at the adapter boundary. Expected resource, timeout,
cancellation, and validation failures do not receive crash tracebacks.
