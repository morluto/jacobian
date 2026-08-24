# Operation references

[Documentation home](../../index.md) · [Tool surface](../tools.md)

The immutable `operation://catalog` resource is the exact bulk inventory for a
server: it contains every operation ID, request schema, result schema, and
example. For the ordinary agent path, use `math.find` to search for an operation,
browse compact sorted operation cards, or inspect one exact operation.

The only operation references maintained outside that live catalog describe
the external boundary that needs extra operational context:

- [SAT and SMT](sat-smt/index.md)
- [Lean source checking](lean/index.md)
- [Exact rational quadratic forms](quadratic-forms.md)
