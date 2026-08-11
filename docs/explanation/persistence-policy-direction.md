# Persistence policy direction (issue #386)

Status: design track for hybrid mathematical intermediate retention.

## Decision needed

Define when Jacobian returns an inline value, a durable mathematical artifact, an
evidence artifact, a verification record, or an execution episode—without
materializing every leaf computation.

## Direction

| Role | Persist when |
| --- | --- |
| Inline mathematical value | Small, one-shot, no downstream binding or replay contract |
| Mathematical artifact | Reuse, composition, exact identity, or later capability binding |
| Evidence artifact | Certificate, witness, search ledger, open obligation |
| Verification record | Authorized checker decision bound to exact claim and evidence |
| Execution episode | Durable “what happened” distinct from the mathematical object |

## Constraints

- Do not weaken checker authorization or content-addressed evidence binding.
- Do not turn computed or search output into `VERIFIED`.
- Do not force a research workflow through persistence policy.
- Prefer domain-owned artifact production over generic `artifact.put` in the
  default agent-facing catalog.
- Measure storage, response size, and task success before removing dual
  representations.

## Next implementation steps

1. Add an explicit persistence intent to capability/operation contracts.
2. Default deterministic leaf ops to inline-only with explicit replay status.
3. Keep durable artifacts for verifier-bound and resumable producers.
4. Evaluation: storage writes, URI reuse, verification correctness before/after.

This note does not change runtime behavior; it records the accepted design
direction for a follow-up implementation PR.
