# Lean source checking

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`lean.check` elaborates one bounded Lean source snippet in the fixed Lean
environment included in the service image. It returns either `ELABORATED` or
`REJECTED` and a bounded list of typed diagnostics.

Each invocation uses a request-scoped temporary directory, removes it when the
process exits, and returns the resulting diagnostics. A timeout or process
failure is an execution failure, not an elaboration result.
