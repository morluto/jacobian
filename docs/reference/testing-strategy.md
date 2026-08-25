# Testing strategy

[Documentation home](../index.md)

Tests prove one observable mathematical or transport contract at a time.

## Routine validation

Run the bounded local handoff before sharing a code change:

```sh
make setup
make check
```

`make check` runs Ruff, mypy, and the math, catalog, dispatch, CLI,
and tooling owners. Add the narrowest named lane when a change crosses its
real boundary:

| Change | Additional check |
| --- | --- |
| MCP tool schema or transport | `make test-mcp` |
| One mathematical domain | `make test-math TESTS=tests/math/logic/test_tools.py` |
| Cross-owner behavior | `make test-integration` |
| Child-process behavior | `make test-process` |
| Singular ideal backend | `make test-singular` |
| Documentation | `make docs-linkcheck` |

For the normal edit loop, run one changed test path through its semantic owner
instead of the broad confidence gate:

```sh
make test-focused LANE=math TESTS=tests/math/graphs/test_graph_distance_matrix.py
```

The explicit lane preserves its configured timeout and worker count. Supported
focused lanes are `math`, `catalog`, `dispatch`, `cli`, `tooling`,
`integration`, `process`, and `mcp`; Singular retains its dedicated command.

`make check-all` is an intentional broad reproduction. Do not use a full suite
as a substitute for a focused regression test.

## CI lifecycle

Pull requests always run static validation. The checked-in CI planner selects
the changed `tests/math` owner and, when a public operation, model, admission,
or canonical contract changes, catalog conformance plus the advertised-example
integration test. Shared runtime, CI, dependency, and unmapped mathematical
paths fail closed to the full math suite. Merge-group candidates and `main`
always run the full non-exhaustive `tests/math` suite. Scheduled validation owns
exhaustive and repeated property checks. Collection inspection remains a manual
diagnostic, not a prerequisite for every test lane.

## What to test

For an operation, test the typed request boundary, the domain result, and a
real caller-visible invocation when the MCP projection changed. The integration
catalog test executes every advertised invocation example. When one result feeds
another operation, serialize the producer result and pass its canonical value
unchanged through the consumer's typed payload. The test should fail if a caller
would have to reconstruct mathematical context or translate between parallel
representations.
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

Examples prove that a documented path works; they do not establish an
algebraic claim. Select evidence by the claim being made:

| Contract claim | Required evidence |
| --- | --- |
| Request domain | Accepted and rejected boundary cases |
| Backend domain | Supported edge and an immediately unsupported case |
| Exact decomposition | Reconstruction property |
| Canonical value | Normalization and round-trip property |
| Parent identity | Incompatible-parent rejection and explicit-map success |
| Algebraic operation | Defining identities or an independent oracle |
| Public operation | Catalog mutation conformance |
| Process backend | Codec, version, timeout/output, and typed failure tests |

For example, an ideal radical needs containment and radicality evidence or an
independent bounded oracle. A factorization needs reconstruction, retained
unit, and positive-multiplicity properties. Known answers remain useful
regressions, but they do not replace these defining properties.

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
