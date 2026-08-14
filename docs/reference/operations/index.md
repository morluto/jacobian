# Operation references

[Documentation home](../../index.md) · [Tool surface](../tools.md)

The immutable `operation://catalog` resource is the exact inventory for a
server: it contains every operation ID, request schema, result schema, and
example. Use `math.find` for a smaller search or one exact inspection.

The only operation references maintained outside that live catalog describe
the external boundary that needs extra operational context:

- [SAT and SMT](sat-smt/index.md)
- [Lean source checking](lean/index.md)
