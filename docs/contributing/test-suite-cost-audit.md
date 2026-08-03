# Test-suite cost audit

> **Historical baseline.** Counts and timings below describe the audit snapshot,
> not current mutable scheduling state. CI timing hints now come from successful
> `main` artifacts and may be absent without affecting test selection.

[Documentation home](../index.md)

This audit records the 2026-07-26 local measurements used to restore a short
edit-test loop without weakening Jacobian's verification boundary. Timings are
machine-local observations, not performance gates.

## 2026-07-28 bootstrap follow-up

A later profile found that runtime construction had regressed to 31.6 seconds
for a fresh core store. The main costs were repeated JSON Schema metaschema
validation and one durable SQLite transaction per descriptor. Bootstrap now
reuses exact-schema validation within the process and installs the capability
portfolio through one store-owned transaction. Ordinary artifact writes retain
their existing durability boundary.

On the same local host, fresh core construction fell from 31.6 to 13.0 seconds.
Attaching to a copied core snapshot took 0.79 seconds, and adding authorized
references to a copied core snapshot took 3.68 seconds. The pre-topology edit
lane fell from 237.91 seconds wall time to 56.55 seconds while selecting 591
tests; three exhaustive cases were excluded from that baseline.
These are single-host observations, not timing gates.

## 2026-07-29 core scheduler follow-up

The pre-topology core lane was benchmarked in three alternating, same-seed pairs with four
workers. Targeted `loadgroup` scheduling, with only the reference claim schema
module grouped around its expensive shared fixture, had a 45.5-second median
wall time. The existing `worksteal` scheduler had a 46.9-second median. Host
noise was material, so this is a small scheduling improvement rather than a
performance guarantee.

That historical `core`/`integration` split has since been replaced by semantic
lanes. The current topology uses sequential unit and storage/Lean lanes,
`worksteal` for component, domain, process, and MCP lanes, and at most two
workers for composition and provider boundaries. A single scheduler cannot
express those incompatible resource limits.

## 2026-07-29 compatibility-lane decision

The exhaustive `main` run
[30453823340](https://github.com/morluto/jacobian/actions/runs/30453823340)
at tree `e4b8fd2673bc7f404edc4c8489dd0f0e4d264071` measured the Python
3.13 compatibility job at 8 minutes 29 seconds. Pytest accounted for 490.16
seconds of that job; checkout and environment setup before pytest took about
11 seconds. The JUnit report attributed 39.8 aggregate case-seconds to the 783
core cases and 914.7 aggregate case-seconds to the 771 integration and
end-to-end cases. Aggregate case time can exceed wall time because xdist runs
cases concurrently.

The same run's four Python 3.12 pre-topology integration shards had a 1.10x max/min wall-time
ratio. Nearby pull-request and post-merge runs measured 1.19x and 1.14x,
respectively, so the duration-fed shard allocation was not the source of the
critical span.

That historical exhaustive compatibility job has been replaced by a small
Python 3.13 import/API compatibility smoke. Full correctness lanes remain on
the canonical Python version; expanding compatibility again requires measured
critical-span and runner-cost evidence.

## Measured lanes

| Lane | Selected tests | Observed wall time | Purpose |
| --- | ---: | ---: | --- |
| Unfiltered `uv run pytest` | 546 | 372.48 s | Diagnostic baseline; mixes Lean into the general xdist pool |
| Pre-topology edit loop, before this audit | 246 | 43.92 s | Non-domain baseline |
| Pre-topology edit loop, after fixture reuse | 246 | 6.55 s | Non-domain baseline |
| Integration, excluding end-to-end and Lean | 275 | 218.88 s | Real stores, subprocesses, adapters, and capability composition |
| End-to-end | 5 | 33.84 s | Distinct complete mathematical workflows |
| `make test-lean` | 20 | 218.26 s | Serial pinned Lean and Mathlib coverage |

Static validation was not a material bottleneck: Ruff, formatting, mypy,
dependency checks, build, and documentation checks completed in about 16
seconds together.

## Findings

Most elapsed time is setup and boundary coverage, not repeated assertions.
Fresh `JacobianRuntime` construction took roughly 2.4 to 3.3 seconds without
reference installation and 3.3 to 4.9 seconds with it. Integration tests
intentionally pay for real schema registration, artifact writes, SQLite and
filesystem behavior, checker subprocesses, plugin isolation, and provider
discovery. The slow Lean cases separately cover Mathlib discovery, real replay,
premise retrieval, declaration indexes, tampering, and evaluation traces.

Those costs protect different invariants. Do not reduce them by globally
disabling durable writes, sharing a mutable runtime between isolation tests,
replacing real filesystems with in-memory substitutes, or deleting
trust-boundary attacks.

One direct DRAT-trim checker module did repeat unrelated product setup. Each
parameter case constructed the full runtime even though it needed only SAT
artifact schemas and a request envelope. A module-scoped minimal real artifact
store now reuses immutable content-addressed artifacts while constructing an
isolated request value for each case. All attack cases remain. This reduced the
fast lane by about 85 percent, from 43.92 to 6.55 seconds.

Pull requests now use separate semantic lanes on the canonical Python version.
Domain shards may use validated `pytest-split` timing; xdist's `worksteal`
balances compatible shards. Storage, process, provider, Lean, and e2e work
retain dedicated resource lanes. The merge queue adds compatibility, coverage,
ordering, and stress evidence without mixing incompatible jobs.

## Development policy

Use the cheapest lane that preserves the boundary being changed:

```sh
make test-unit
make test-component TESTS=tests/component/capabilities/test_mcp_invocation_projection.py
make test-composition
make test-e2e
make test-lean
make test-all-ci
```

Use focused component, domain, composition, or boundary tests while changing
stores, adapters, plugins, subprocesses, or checker execution. `make check`
combines Ruff, strict typing, and the unit lane for the routine local handoff.
Dependency and dead-code analysis and package builds remain available through
`make check-static` but are CI-owned rather than routine local handoff work.
The installed pre-push hook runs only `make lint typecheck`; affected tests and
CI own correctness lanes. `make test-all-ci` is the explicit exceptional local
reproduction of every semantic lane. Python 3.13 compatibility, combined
coverage, security, duplicate-code, and npm validation remain separate CI
lanes. Do not use unfiltered `uv run pytest` as the default handoff command:
semantic lanes keep incompatible resources in separate invocations.

The Lean suite runs serially on one prepared runner. This avoids concurrent
multi-gigabyte Mathlib processes and keeps the pinned toolchain setup attached
to the tests that consume it.

## Follow-up opportunities

- Profile runtime startup as a product concern before changing its durability
  or registration model.
- Reuse module fixtures only where inputs remain isolated and the shared state
  is immutable.
- Track lane wall times periodically; investigate changes before adding a
  blanket cost marker or weakening required coverage.
- Move backend combinations to a slower lane only when the pull-request lane
  still exercises every affected trust boundary.

## Follow-up audit

The expanded 537-test non-Lean suite has a median recorded case duration of
0.08 seconds, but its slowest five percent take at least 6.62 seconds. Those
cases are not interchangeable repetitions: they cover remote tenant isolation,
MCP and CLI process boundaries, interrupted-search recovery, clean-process
checker replay, SAT proof interoperability, and complete end-to-end workflows.
The two supported Python versions exercise runtime compatibility, while the
Lean lane exercises a separate pinned toolchain and checker boundary. Retain
those lanes.

An exploratory eight-worker run on an eight-logical-CPU, 32 GB Linux host
completed all 537 tests in 137.81 seconds, compared with about 170 seconds under
the four-worker default. This single-host result is not sufficient to raise the
default: the suite is subprocess-heavy, wall time changed substantially under
unrelated host load, and exhaustive local validation is deliberately not the
routine loop. Keep the stable four-worker cap and revisit it only with
controlled repeated measurements on local and CI runners.

The actionable redundancy was procedural. `make check` now combines Ruff,
mypy, and the unit lane, while semantic targets expose focused component,
domain, composition, storage, process, MCP, provider, Lean, and e2e checks. CI
skips heavy lanes for documentation-only and npm-only changes, uses explicit
suite ownership for known paths, and fails closed for unknown ones. Focused
Python and Lean debug workflows provide remote reproduction without rerunning
unrelated matrices. On the measured host, the resulting `make check` completed
256 selected tests in 8.36 seconds.

Source-to-suite impact is declared in `.github/ci-impact.json` and tested
against tracked source files. Unknown paths still fail closed. Each measured CI
run reports its critical span, summed runner minutes, and longest job, making
both reviewer latency and compute growth visible. The critical span is the
interval from the earliest reported job start to the latest reported job end;
it is not a dependency-graph reconstruction or the full workflow elapsed time.
Scheduled lanes exercise repeated property tests, alternate orders, and
optional providers outside the pull-request critical span.

Do not run every semantic lane repeatedly during implementation and then
immediately repeat them in pull-request CI. Use `make check` plus the affected
focused target, and let CI provide one exhaustive pass on the final tree. Run
`make test-all-ci` locally only when CI is unavailable or an
environment-specific failure needs reproduction.

Some short CI jobs still overlap in setup or packaging work, but they run in
parallel and were not on the measured critical span. Consolidating them would
increase workflow coupling without materially shortening feedback, so this
audit leaves them unchanged.

Ephemeral timing artifacts feed the domain and composition shards. Successful
`main` runs publish fresh history; missing or invalid history falls back to equal
weighting. Storage, process, provider, Lean, and e2e lanes retain their own
resource topology rather than being mixed to equalize aggregate duration.

Do not stack a local duration refresh, full integration profiling, and focused
module debugging on the same host at once. That contention recreates the
pull-request wall-time problem the lane split exists to avoid: routine
`make check`, exhaustive merge-queue validation, and scheduled
stress work must remain separate executions.

The fixture boundary is now explicit. Unit tests receive value fixtures only;
component tests own one real service; domain tests receive `domain_services`
with explicit bundles; composition tests opt into visibly expensive
`fresh_complete_runtime`, `attached_complete_runtime`, or
`authorized_complete_runtime`; and boundary tests own durable stores,
checker/process servers, MCP transports, and provider environments. Immutable
templates are built in a temporary sibling and atomically renamed before tests
copy their own state. No mutable runtime, registry, or connection is
shared between tests.


## 2026-07-29 topology cost controls

The semantic lanes make the resource boundary explicit:

- Python 3.13 runs a small import/API compatibility smoke instead of duplicating
  the complete suite.
- Makefile edits no longer route to unrelated Lean or npm jobs.
- The pre-push hook runs only Ruff and mypy; affected tests and CI own
  correctness lanes.
- Scheduled benchmark comparisons use a 25% noise allowance and fail on a
  classified regression.
- The global signal timeout is gone. Lane runners supply deadlines, and native
  or external work belongs in killable process boundaries.
- Stress selects the explicitly marked property tests; its repetition count is
  visible as STRESS_COUNT.

These controls complement the historical measurements above. They do not turn
machine-local timing into a correctness assertion.

## 2026-08-03 hotspot remediation follow-up

A churn audit found remaining DX bottlenecks after the topology split and the
agent-workflow per-task leaf migration (#376):

1. **Unused complete-runtime `usefixtures` marks.** Several composition,
   storage, and provider modules forced `attached_complete_runtime` or
   `authorized_complete_runtime` construction even when tests never referenced
   those fixtures. Search orchestration double-paid unused attach plus
   `fresh_complete_runtime`. On this host, `create_runtime` from an empty state
   measured about 4.9 seconds. Removing the unused marks kept all assertions
   and cut `test_graph_composition.py` from **19.7s to 14.0s** wall time
   (same 11 cases). Search orchestration wall time stayed noise-bound (~66s)
   because fresh construction dominates; per-case setup dropped about 0.8s when
   the unused attach was removed.

2. **Harbor validation serial pytest.** `make harbor-validate` collected about
   1096 tests under `benchmarks/validation`. Path monkeypatches in verifier
   helpers are process-local and restored in `finally`, so modest xdist is safe.
   Default is now `HARBOR_VALIDATION_WORKERS=2` (~**14s → 8s** on this host).
   Oracle and adapter Make targets remain serial. The hardcoded `gap.json`
   count (merge magnet on every task add) was replaced by a non-empty +
   historical-provenance check.

3. **Mega-module merge magnets** were split without dropping cases:
   exact-domain checkers by domain, observation-results by concern, capability
   service by concern, and search orchestration into lifecycle / recovery /
   plugin-fail-closed leaves.

4. **CI over-selection.** `Makefile` edits now select only `static` + `build`;
   lane topology changes live under `tools/**` and `tests/topology.toml`
   (`test-topology-runners`). Domain packages under `src/jacobian/domains/**`
   select unit + component + domain + static + build (not storage/mcp/e2e) via
   `domain-mathematical-sources`, which suppresses the general `python-source`
   catch-all when both match. Verification-boundary, packaging, and fallback
   remain fail-closed.

These are single-host observations, not timing gates. Do not recreate shared
`test_task_regressions_*.py` dumps; keep Harbor attack coverage in per-task
leaves plus `test_generic_verifier_contracts.py`.
