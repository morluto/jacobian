# Benchmark contracts

[Benchmark home](../README.md)

Harbor benchmarks are external evaluation assets. A task owns its
agent-visible instruction, hidden Oracle material, verifier, pinned environment,
and task digest under `benchmarks/datasets/`. Jacobian is an experimental
treatment the benchmark may expose; benchmark design does not extend Jacobian's
public API.

Choose tasks for the mathematical capability they measure, including
capabilities the current operation library cannot solve. Freeze task and
environment identity, keep hidden material out of the agent environment, and
report limitations with the result.

## Public submission contract

For an atomic mathematical task, reward is normally binary: `1` only when the
verifier can replay the submitted result against the frozen input and every
declared witness condition holds; otherwise `0`. Tool calls, prose, confidence,
and diagnostic observations do not earn credit.

The public submission is normally `{ "result": ... }`. Add `"witness"` only
for a finite task-specific mathematical object that replay cannot derive from
the input and result. Do not require duplicate result files, prose
explanations, leaked conclusions, or verifier-derived status fields.

The instruction, submission schema, hidden solver, and verifier must accept the
same objects. When a schema changes, update all four together plus the gold
submission and public-contract fixture.

## Replay authority

Correctness comes from task-local replay using the frozen verifier copy of
`input.json` and the submitted mathematical value. `tests/expected.json` is an
Oracle regression fixture, not the definition of correctness. Changing only an
expected fixture must not change reward for a fixed input/submission pair.

Independent claims must be checked independently. A corrupted field or failed
subclaim must not decide an unrelated collision, invertibility, completeness,
or witness claim.

## Mathematical representations

Compare represented mathematical values rather than preferred renderings:

- parse structured rationals as exact fractions;
- treat sets, maps, distributions, and sparse polynomials as unordered unless
  order is mathematically part of the task;
- represent formulas with the smallest task-owned structure rather than scored
  prose;
- require canonical form only when canonicalization is itself a stated outcome.

Reject malformed types, booleans used as integers, invalid denominators,
non-finite values, and resource-bound violations at the boundary.

## Witnesses and artifacts

`answer.txt` is not an authoritative submission interface. A witness artifact
is justified only when replay needs an external finite object; it must not
mirror or hash `result`, carry boilerplate prose, or exist merely because an
older task used one. Gold witness paths must resolve to regular files beneath
the task's `solution/` root.

Generated task families should expose only the certificate variants their
public claim can reward. Do not publish one universal certificate union across
unrelated claim families.

## Evaluation output

Task bundles and immutable snapshot locks are reproducibility inputs. Run
outputs, trajectories, reports, and other regenerable evidence belong under
ignored `benchmarks/results/` or external artifact storage. A benchmark result
is evidence about the experiment; it is not a new mathematical or product
contract for Jacobian.
