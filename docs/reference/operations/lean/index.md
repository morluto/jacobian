# Lean source checking

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`lean.check` elaborates one bounded Lean source snippet in the fixed Lean
environment included in the service image. It returns either `ELABORATED` or
`REJECTED` and a bounded list of typed diagnostics.

The operation uses a temporary directory for that one invocation and removes it
when the process exits. It has no proof-state session, declaration search,
cache, source publication, replay record, or stored proof reference. A timeout
or process failure is an execution failure, not an elaboration result.
