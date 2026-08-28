# Testing strategy

[Documentation home](../index.md)

Tests prove one observable mathematical or transport contract at a time.

## Routine validation

Run planner-selected affected validation before sharing a code change:

```sh
make setup
make affected AFFECTED_BASE=origin/main
```

`make affected` resolves `AFFECTED_BASE...HEAD` together with staged, unstaged, and
untracked paths, then uses the checked-in pull-request planner to run only the
selected owner, catalog, and boundary lanes. Ruff and mypy run only on changed
Python files in their normal repository scopes. It
prints the immutable plan and its reasons before execution; `make affected-plan`
prints that selection without running it. Installed-wheel evidence remains
CI-owned. For a deliberately narrower one-owner loop, use `make handoff-scoped`
with explicit `PATHS`. Add the narrowest named boundary lane when a change
crosses its real boundary:

| Change | Additional check |
| --- | --- |
| MCP tool schema or transport | `make test-mcp` |
| One mathematical domain | `make handoff LANE=math TESTS=tests/math/logic/test_tools.py` |
| Cross-owner behavior | `make test-integration` |
| Child-process behavior | `make test-process` |
| Singular ideal backend | `make test-singular` |
| Documentation | `make docs-linkcheck` |

`make quick LANE=... TESTS=...` omits mypy but retains the repository-wide Ruff
check. In a shared checkout, use `make quick-scoped LANE=... TESTS=...
PATHS="src/... tests/..."` to scope both Ruff and the test path. The explicit
lane preserves its configured timeout and worker count. Supported
focused lanes are `math`, `catalog`, `dispatch`, `cli`, `tooling`,
`integration`, `process`, and `mcp`; Singular retains its dedicated command.

In a shared checkout with unrelated static drift, declare the source and test
paths you own instead of waiting on unrelated files:

```sh
make handoff-scoped \
  LANE=math \
  TESTS=tests/math/graphs/test_graph_distance_matrix.py \
  PATHS="src/jacobian/math/graphs tests/math/graphs/test_graph_distance_matrix.py"
```

`PATHS` is required and is passed directly to Ruff and mypy; use repository
paths only, without pytest node IDs. This is additive local evidence, not a
replacement for `make handoff`, `make check`, or CI's full static gate.

`make check` is the final broad ordinary gate: it runs lint, types, and all
non-integration owners once, excluding separately marked property, exhaustive,
and scale evidence. `make check-all` adds the ordinary integration owner.
`make test-full` is the exceptional complete local reproduction. Do not use a
broad or full suite as a substitute for a focused regression test.

The scale lane defaults to two workers because its exact boundary cases can
consume substantially more memory than ordinary tests. Hosted CI explicitly
sets `SCALE_WORKERS=4`; contributors can keep the default or override it for a
host with less capacity.

### Command hierarchy and timing evidence

Use `make affected` for normal branch-local validation and
`make handoff-scoped LANE=... TESTS=... PATHS="..."` for a one-owner edit loop.
Run `make check` once on a frozen tree. `make check-all` reproduces ordinary CI
lanes and `make test-full` is the complete local escalation path; neither is an
iteration command.

`make check` and `make check-all` take a worktree-local non-blocking validation
lease. `make validation-status` identifies a competing broad run immediately;
focused and affected commands remain unblocked. Use
`ALLOW_PARALLEL_VALIDATION=1` only for an intentional parallel broad run on a
host with known capacity.

Every CI lane emits a JUnit artifact and a worker-timing sidecar retained for
90 days. Download both to identify slow testcases, xdist call-time skew, and
the non-call wall remainder before changing worker counts, fixtures, or shards:

```sh
make test-timings JUNIT=pytest.xml TIMING=timing.json
```

## CI lifecycle

Pull requests always run static validation. The checked-in CI planner selects
changed mathematical owners and changed dispatch, CLI, tooling, integration,
process, MCP, Singular, and installed-wheel boundaries. A public operation,
model, admission, or canonical contract change also selects catalog conformance
and the advertised-example integration test. Shared runtime, CI, dependency,
and unmapped paths fail closed to the complete ordinary suite.

Merge-group candidates run the complete ordinary suite plus optional
near-envelope scale evidence. `main` runs the landed-tree ordinary suite and
coverage. Scheduled validation owns exhaustive evidence, repeated property
checks, optional scale evidence, and randomized order/provider variation; the
deferred exhaustive and scale lanes run as independent jobs so they do not
serialize one another.
Each ordinary CI lane retains JUnit timing evidence for 90 days. Use those node
durations to tier near-envelope regressions before proposing more workers or
timing-based sharding; ordinary product lanes remain unsharded until the
evidence demonstrates a safe, useful split. Collection inspection remains a
manual diagnostic, not a prerequisite for every test lane.

Markers are execution tiers, not synonyms for slow tests:

| Marker | Evidence | Owner |
| --- | --- | --- |
| `property` | Repeated or generalized invariant checks | Scheduled validation; one run in `make test-full` |
| `exhaustive` | Broad finite reference sweeps | Scheduled validation; one run in `make test-full` |
| `scale` | Optional exact near-envelope execution proof | Merge-group and scheduled validation; one run in `make test-full` |

Keep a small ordinary regression for the same public behavior when moving a
near-envelope case to `scale`.

## What to test

For an operation, test the typed request boundary, the domain result, and a
real caller-visible invocation when the MCP projection changed. The integration
catalog test executes every advertised invocation example. When one result feeds
another operation, serialize the producer result and pass its canonical value
unchanged through the consumer's typed payload. The test should fail if a caller
would have to reconstruct mathematical context or translate between parallel
representations.

When a consumer relies on a theorem-bearing subtype, add an adversarial type-
boundary fixture: construct a value that is structurally valid but violates the
required theorem property, and prove that the consumer rejects it or returns
the declared typed non-applicability outcome. Pair that fixture with a positive
recognizer → serialization → consumer test using the canonical validated
subtype unchanged. Shape validation and a separate recognizer test do not by
themselves establish safe composition.

Also submit a forged serialized subtype directly to the public consumer without
calling its recognizer first. The test must prove that public reconstruction
revalidates the theorem property, checks source-bound evidence through the
declared bounded verifier, or routes through consumer-owned recognition. A
nominal subtype tag, `validated` field, or producer-shaped payload must not
cross the stateless boundary as proof.

When an operation is added or changed because of a source-backed gap, preserve
at least one minimally reduced motivating request as a behavioral regression.
Run it through the final public request model and operation, and assert both
admission and the typed mathematical result. A smaller happy-path example does
not replace the request that established the need.
Include the degenerate producer case most likely to erase ambient information,
such as an empty basis, zero-row matrix, empty trace, or zero count. For
source-bound decisions whose public contract accepts independently supplied
conclusions or certificates, mutate the source and conclusion independently
and require the bounded result-verification path to reject both forgeries. For
ordinary computed results, test the defining invariant of the returned value;
do not introduce a replay path solely for that test.

For every exact-success branch, assert the defining equation or preservation
law and reject every representable status/diagnostic combination that would
claim success while admitting that the invariant failed. Checking only that a
step list, factor list, witness, or certificate-shaped object is nonempty is
not correctness evidence. For non-success branches, assert that exact witness
fields are absent unless the contract explicitly assigns them another meaning.
When branches have different fields or field meanings, inspect the generated
schema for a discriminated union and validate every branch through both the
declared result type and final transport projection. Construct contradictory
payloads directly; proving only that the producer does not emit them is
insufficient.

Center correctness tests on regression evidence when fixing a concrete
regression, defining identities, consistency checks, and property-based tests
against actual symbolic or mathematical behavior. Use maintained libraries in
their owning domain tests rather than mocking their algorithms. Exercise real
mathematical and numerical assertions at public typed interfaces with
deterministic inputs; keep fixtures sparse and name the evidence
each one contributes. Do not monkeypatch or use test fakes for mathematical
values, validators, serializers, or backend correctness. Keep
unavailable-environment and transport-failure tests in separate boundary
cases; they must not stand in for mathematical correctness. A timeout,
cancellation, unavailable external executable, or solver `UNKNOWN` is never a
positive mathematical conclusion.

Boundary rejection tests do not replace a realistic admitted workload. For a
new or repaired operation, run at least one motivating interior request through
the final public boundary and assert its typed mathematical result. When the
operation has multiple mandatory phases, test that they share one deadline and
work ledger rather than receiving independent wall budgets. Test caller-side
timeouts separately in the MCP lane; their only conclusion is that transport
aborted the request.

Dispatch rejection tests must name the boundary they exercise. A malformed or
structurally invalid payload raises `OperationRequestValidationError`. A
structurally valid payload rejected by the native mathematical or resource
admission raises `OperationDomainValidationError`; where practical, assert that
native and dispatch calls preserve the same structured owner error code. MCP
projects both validation classes as `INVALID_PARAMS`, while unexpected backend
failures and timeouts remain operational errors. Do not update a semantic
admission test to expect request-model validation merely because both failures
appear as an invalid-parameter response over MCP.

For an operation that uses a nontrivial backend, enumeration, solver, or
certificate check, add the smallest owner-local regression that proves a
successful producer performs that work once. If independently supplied claims
are supported, separately prove a forged claim fails through the explicit
verifier. Timing tests must state whether serialization is included in the
reported duration or exposed as a distinct named phase; no mandatory phase may
be absent from all timing evidence.

Examples prove that a documented path works; they do not establish an
algebraic claim. Select evidence by the claim being made:

| Contract claim | Required evidence |
| --- | --- |
| Request domain | Accepted and rejected boundary cases |
| Backend domain | Supported edge and an immediately unsupported case |
| Exact decomposition | Reconstruction property |
| Canonical value | Normalization and round-trip property |
| Theorem-bearing subtype | Invalid structural candidate and forged validated payload rejected, then recognizer → serialization → consumer success |
| Exact-success state | Defining invariant holds, schema exposes discriminated branches, and contradictory combinations are rejected |
| Parent identity | Incompatible-parent rejection and explicit-map success |
| Algebraic operation | Defining identities or an independent oracle |
| Public operation | Catalog mutation conformance |
| Process backend | Codec, version, timeout/output, and typed failure tests |

When correcting an admission or boundedness rule, inspect the affected owner
lane for assertions that pin the previous accept/reject outcome and rewrite or
remove them in the same change. Every retained rejected boundary case must
fail for the guard it names: keep the rest of its payload valid and assert the
owner's structured error code. A request rejected by an earlier field bound is
not evidence that a later cross-field admission guard ran.

For example, an ideal radical needs containment and radicality evidence or an
independent bounded oracle. A factorization needs reconstruction, retained
unit, and positive-multiplicity properties. Known answers remain useful
regressions, but they do not replace these defining properties.

### Derived contract bounds in match strings

When an assertion matches a validation message that carries a computed bound
(such as a digit limit, byte budget, or combinatorial ceiling), import the
owning constant or helper from source and build the expected string from it
rather than hardcoding the numeric value. A test that writes `match="10-digit
bound"` will break with an opaque regex mismatch whenever a scale-cap commit on
main raises the bound — even on an unrelated open branch. Instead, import the
constant and interpolate:

```python
from jacobian.math.number_theory.diophantine_approximation._models import (
    _convergent_component_digit_cap,
)

cap = _convergent_component_digit_cap(4)
with pytest.raises(ValueError, match=rf"{cap}-digit bound"):
    ...
```

The message text stays pinned; only the number is derived from the same source
the production code uses. Boundary test inputs (the value that triggers the
error) should likewise be computed from the constant, not hardcoded:

```python
beyond = "9" * (_MAX_MULTIVARIATE_COEFFICIENT_DIGITS + 1)
with pytest.raises(
    ValueError, match=rf"{_MAX_MULTIVARIATE_COEFFICIENT_DIGITS}-digit bound"
):
    ...
```

When the owning constant is private (underscore-prefixed), importing it from
its owning module is acceptable in tests — the test and the source share one
definition.

### Evidence plans for exact operations


Before implementing an exact decomposition, certificate, or authoritative
derived value, state its defining invariant: the reconstruction equation,
preservation law, or independently checkable property that makes a returned
value mathematically valid. Then name the smallest set of tests that establishes
that invariant and rejects plausible results that satisfy only a weaker
mathematical claim.

When authoring or changing an operation's contract or backend adapter, derive
at least one expected value from the mathematical definition rather than from
the maintained backend before merging: a worked computation, brute-force
enumeration on a small admitted slice, or an independently constructed object
such as a determinant assembled directly from coefficients. Agreement with the
backend is not independent evidence — every code path in one backend can share
one defect. Where the operation already advertises an identity, such as
reconstruction, a swap law, or invariance under a change of basis, test that
identity against the same change. Before trusting backend output for a new
claim, consult the [known backend defects](backend-known-defects.md) registry;
add an entry whenever an adapter compensates for backend behavior.

Classify each fixture by the evidence it contributes:

| Fixture role | What it establishes |
| --- | --- |
| Defining-invariant | Tests the reconstruction equation, preservation law, or certificate relation owned by the result; use bounded replay only when the public contract accepts independently supplied result data. |
| Convention/known-answer | Fixes terminology, normalization, indexing, signs, or another convention-sensitive value. |
| Adversarial weaker-semantics | Rejects a tempting result that has the right shape or satisfies only a weaker claim. |
| Metamorphic or equivalence | Checks invariance under a meaning-preserving transformation or compares noncanonical outputs by mathematical equivalence. |
| Producer-consumer | Supplies one operation's serialized result unchanged to a downstream operation and checks the composed meaning. |
| Scale/stress | Exercises a source-backed size or representation regime near a justified admission boundary. |

A fixture may fill more than one role, but the evidence plan should name those
roles rather than treating fixture count as coverage. A large certificate
should normally yield a small discriminating CI fixture, with the full
source-scale case retained only as an optional stress or benchmark fixture when
it remains reproducible and useful. Agreement on a scalar summary is
insufficient when correctness concerns a set, partition, or family of canonical
representatives: compare the exact objects or their declared equivalence class.
For example, two tree generators can return the same count while duplicating
one representative and omitting another.

Accepted and rejected boundary cases prove that a stated envelope is enforced;
they do not prove that the envelope admits realistic work. Pair boundary tests
with a source-backed accepted case even when its public dimensions are small.
A compact request can still be the decisive scale fixture when private
normalization, expansion, or backend representation makes it expensive.

Use a source-backed reference fixture when a standard example helps fix
terminology, normalization, or another convention-sensitive output. Cite the
specific theorem or example and record the convention the fixture depends on.
Pair that fixture with the relevant boundary and adversarial cases and with a
reconstruction, defining-identity, bounded exhaustive, or independent-oracle
test. A reference fixture is an anchor, not the correctness argument.

When a valid result is not unique, compare its mathematical equivalence class
or validate its reconstruction. Do not require incidental backend ordering,
temporary identifiers, tree roots, or one particular witness unless the public
contract makes that choice canonical.

Singular testing follows the same ownership split as other child-process
backends. The shared bounded-process supervisor owns process-group termination,
including descendants that ignore termination or retain inherited pipes. The
Singular lane tests only adapter-specific behavior: timeout and execution-outcome
projection, supported-version enforcement, strict codec behavior, output limits,
and request-scoped cleanup. Do not duplicate the supervisor's termination suite
for each mathematical backend.

The commutative-algebra domain checks Singular's mathematical results against an
independent combinatorial oracle on bounded monomial ideals, in addition to
containment, idempotence, identity, and colon-law properties. SageMath may be
used as a development-time differential oracle, but it is not a runtime or
required-CI dependency.

## Documentation acceptance

Documentation should describe current behavior rather than refactor history.
Link to the [product blueprint](../explanation/product-blueprint.md) for product
philosophy, and keep tool/reference pages focused on the contracts they own.

Run `make docs-linkcheck` after changing Markdown. It validates relative links,
documented Make commands, and documented test paths.
