# Durable search capability runtime

[Documentation home](../index.md)

- Status: Architecture note

This document describes the operational design available to long-running search
capability adapters. Mathematical search semantics remain in domain plugins;
the runtime owns request acceptance, lifecycle state, persistence, accounting,
and routing to authorized verification services. It is execution and trust
policy, not mathematical strategy. The runtime does not prescribe how an agent
composes search, evaluation, transformation, or verification capabilities, and
search output never self-certifies.

This runtime is not a separate public MCP workflow. Agents discover the
namespaced capability with `math.find`, invoke it with
`math.run`, and inspect returned experiment and artifact resources.

## Ownership model

Use one active Jacobian process per state directory. SQLite serializes
transactions inside that process and makes request acceptance durable, but the
current implementation has no lease protocol for multiple coordinators.

The scheduler accepts `workers = 1`. Values above one fail validation. Adapter
operations execute in bounded child processes; the coordinator, lifecycle
threads, and SQLite connection remain local.

## Request acceptance

An invocation's idempotency key identifies one exact canonical request:

```text
new key + request digest
    → one experiment URI

same key + same digest
    → reuse that experiment URI and append REQUEST_REUSED

same key + different digest
    → reject before execution
```

Acceptance uses a SQLite `BEGIN IMMEDIATE` transaction. Concurrent submissions
cannot create two accepted experiments for the same key. The accepted snapshot
binds the claim, plugin manifest and registry snapshot, adapter and dependency
implementation digests, effective budget, and environment digest.

## Lifecycle

```text
PENDING ──► RUNNING ──► COMPLETED
   │           │  ├──► TIMEOUT
   │           │  ├──► ERROR
   │           │  ├──► PAUSE_REQUESTED ──► PAUSED ──► PENDING
   │           │  └──► CANCEL_REQUESTED ──► CANCELLED
   └───────────┴─────► CANCEL_REQUESTED
```

Pause takes effect at a checkpoint boundary. Resume continues the same
experiment URI, idempotency binding, archive lineage, and accounting. Cancelled
or timed-out work keeps all committed artifacts and events.

Operational state never supplies a mathematical conclusion. In particular,
`COMPLETED` means only that the adapter stopped.

## Durable data

The runtime combines a small mutable index with immutable evidence:

| Data | Storage | Role |
| --- | --- | --- |
| Latest experiment snapshot | `search_experiments` SQLite row | Current lifecycle index |
| Idempotency binding | `search_idempotency` SQLite row | One key/request/experiment mapping |
| Lifecycle events | Append-only `search_events` rows | Ordered request, operation, control, retry, and recovery history |
| Recovery failures | `search_recovery_failures` SQLite row | Quarantine evidence for one malformed snapshot |
| Archive page | Immutable artifact | Candidate, evaluation, witness, and result records for one iteration |
| Checkpoint | Immutable artifact | Opaque adapter state plus identity, budget, accounting, and prior-checkpoint link |
| Terminal archive | Immutable artifact | Root for the completed committed lineage |

SQLite triggers reject lifecycle-event updates and deletes. Event digests bind
the predecessor so readers can validate the chain.

## Checkpoint commit boundary

For each successful iteration, the runtime:

1. stores candidates, evaluations, witnesses, and results;
2. stores an immutable archive page;
3. stores an immutable checkpoint;
4. samples wall time;
5. commits the new snapshot and lifecycle event in SQLite.

The wall-time sample includes checkpoint artifact creation and excludes the
following metadata transaction. If artifact persistence crosses the wall
budget, the experiment becomes `TIMEOUT` even when the adapter reported
completion.

Operation counts are exact for committed lineage. Wall time is measured at the
boundary above. The current runtime does not claim CPU, memory, network-byte,
or filesystem-byte metering.

## Restart recovery

Runtime startup examines recoverable, cancelled, and unknown indexed states in
one transaction:

- `PENDING`, `RUNNING`, and `PAUSE_REQUESTED` become `PAUSED`;
- `CANCEL_REQUESTED` becomes `CANCELLED`;
- a cancelled run missing its terminal archive receives one;
- malformed JSON, an invalid indexed state, or a row/snapshot identity or state
  mismatch is quarantined as `ERROR`.

Quarantine records a digest of the stored bytes and a failure detail, then
continues with the next row. One corrupt invocation therefore cannot prevent
unrelated recovery.

Resume validates the checkpoint and archive pages before accepting opaque
plugin state. The request digest, experiment URI, registry snapshot,
implementation digests, environment, effective budget, accounting, and page
lineage must all agree.

Work performed after the last committed checkpoint may execute again after a
crash. It does not create a second accepted experiment, and only a committed
page or checkpoint becomes durable lineage.

## Authority and containment

Effective Jacobian-represented authority is the restrictive intersection of
the plugin contract, operator policy, and invocation request. A plugin or
request cannot widen budgets, capability selection, artifact bindings, or
checker authorization.

Child-process time and output limits are operational controls, not a security
sandbox. Workers inherit the coordinator's network and filesystem boundary.
Run Jacobian under an OS or container policy when operator-installed code needs
stronger isolation.

## Related references

- [Plugin conformance kit](../reference/plugin-conformance.md)
- [Testing strategy](../reference/testing-strategy.md)
