# Domain operation library

Every built-in operation is a direct typed mathematical function with one
domain owner. Declaration modules export immutable tuples of
`MathTool` values. `math.find` reads those entries and `math.run`
validates then executes exactly one of them.

The canonical operation path and ownership boundaries are defined in the
[architecture](../explanation/architecture.md). A domain function may use a
maintained library privately for its algorithm;
callers see Jacobian's typed mathematical values, not backend objects. The
execution plan is request-scoped and internal. It is not caller-owned workflow
state and does not add another MCP operation.

Keep values, codecs, invariants, and backend conversions with their domain.
Shared contracts are limited to passive cross-domain primitives. A bounded
operation reports mathematical completeness or uncertainty in its own result.

Public request contracts must make their valid representation visible before a
backend call. Express constraints that JSON Schema can represent in typed field
metadata. When a domain invariant needs a Pydantic model validator—such as a
cross-field relation or canonical term ordering—also provide an explicit field
or model description and a minimal valid example in the exported schema. The
Pydantic model is authoritative for the serialized request contract; the
metadata lets a caller form a valid first request rather than discover the rule
only through a rejected call. Mathematical and execution semantics belong to
one domain-owned admission function. A wire model may invoke that shared
function after parsing, but must not duplicate it or make the native API build a
wire model just to inherit its checks.

Every built-in `MathTool` declaration must publish at least one small valid
invocation example. An example is part of the public contract: it must validate
against the declaration's request model, use canonical values where required,
and be executable in Jacobian's supported local environment. Keep it close to
the operation and adapt it when a request contract changes. The composition
catalog test executes every published example: the payload must validate, the
domain function must return its declared typed result, and the final transport
projection must accept its canonical representation.
The operation's owning tests still own nontrivial example behavior and the
adversarial and request-boundary cases in the contract review below. Those two cases
are written with the operation; they are not a separate CI program.

Write an invocation example's description in two parts: first state the
computation the operation performs on the supplied values, then state the
important precondition that makes the example valid. For example, use
`Compute the exact eigenvalues of [[1, 2], [3, 4]]; the matrix must be square.`
The first part tells an agent what the operation does; the
second part teaches the input rule it must preserve. A precondition by itself,
such as `The matrix must be square`, is not an adequate example description.

Examples help an agent form its first request without guessing. JSON Schema can
name fields and simple bounds, but it cannot fully communicate validator-owned
rules such as nested value shape, canonical ordering, coupled fields, or the
smallest useful composition. On exact inspection, an agent can copy an example
payload and adapt its mathematical content instead of discovering that wire
contract through a failed call, a lengthy ad-hoc script, or trial and error.
Examples illustrate a valid representation; they do not prescribe a proof
strategy or restrict how operations may be composed.

Avoid **validator-only public contracts**. Do not introduce a required input
representation solely through a Pydantic validator and expect callers to infer
it from an error. When a rule cannot be expressed as an ordinary JSON Schema
constraint, pair the validator with schema-visible field or model guidance and
a valid invocation example. Diagnostics remain the recovery path for malformed
requests; they are not the primary documentation for an operation's wire
contract.

Use maintained backends through thin private adapters. Direct bounded results
compose by being supplied as the next operation's typed payload.

The logic family illustrates the boundary. `sat.cnf.canonicalize` returns a
canonical CNF value; `sat.assignment.check` and `sat.solve` accept that value
directly. `smt.solve` accepts one bounded QF SMT-LIB query.

## Implementation selection

Choose the smallest operational surface that can establish the admitted
mathematical postcondition within its public bounds. Assess implementation
options in this order:

1. Use a maintained in-process Python backend when it supports the complete
   bounded claim.
2. Use a thin native binding when its build, packaging, platform, and runtime
   costs are proportionate to the admitted domain.
3. Use a Jacobian-owned bounded implementation when no lightweight maintained
   backend fits, the admitted bounds make a simpler published algorithm
   practical, the result has a complete reconstruction or defining invariant,
   and ordinary repository tests can establish correctness independently. A
   mature external implementation may provide additional differential evidence.
4. Use a child process only for a concrete isolation, killability, or fixed
   toolchain reason, with the complete process-boundary obligations described in
   the [mathematical backend contract](mathematical-backends.md).
5. Narrow or reject the operation when none of these options can support its
   public claim and work bounds.

Preferring maintained backends does not require importing an entire
cross-language ecosystem when its build, ABI, installation, runtime, or failure
surface is disproportionate to the bounded mathematical kernel Jacobian needs.
The issue or pull request must record why the selected implementation class is
proportionate; backend convenience alone does not justify a broader public
contract.

## Operation contract review

First diagnose the gap: a missing operation is only one of several possible
responses to an observed composition failure. Classify the failure as
representation, interoperability, discovery, contract, scale/backend, operation,
or reasoning before designing an implementation (see
[Executable mathematical vocabulary](../explanation/executable-mathematical-vocabulary.md)).
Only a genuine operation gap proceeds to the catalog-admission contract in
[public-operation-admission](public-operation-admission.md). Existing
operations with request, result, backend, or transport-bound changes still
need the runtime ownership and boundedness review below.

Before trusting backend output for a new claim, consult the
[known backend defects](backend-known-defects.md) registry; add an entry
whenever the adapter compensates for backend behavior instead of narrowing
the public domain.

Every new public operation, or change to a request, result, backend, or
transport contract, records the decisions that apply in its issue or pull
request. Do not fill a universal form: link to the relevant conditional section
below for subtype trust, boundedness, a backend adapter, or a process boundary.

### Public operation contract

- Semantic mathematical domain and postcondition:
- Canonical public value type:
- Result type:
- Reconstruction or defining invariant:
- Request representation, admission quantities, and degenerate cases:
- Result-size bound and exact-success/non-success states:
- Evidence: defining invariant plus the changed boundary regression:

### Validated mathematical subtypes and exact-success states

A structurally valid candidate is not a theorem-bearing value. A raw set
system, matrix, graph, polynomial, or similar carrier establishes only the
invariants its canonical type actually validates. An operation whose result
depends on a stronger property—such as the greedoid axioms, antimatroid union
closure, irreducibility, or stochasticity—must either:

- consume a canonical subtype whose public construction establishes that
  property within the admitted bound; or
- perform recognition as part of its own postcondition and return a typed
  negative or non-applicable outcome when recognition fails.

Do not let a structurally valid candidate flow into a theorem-dependent
consumer merely because its fields have the expected shape. A positive
recognizer should return a source-bound canonical validated value that
downstream consumers accept unchanged. If an operation intentionally computes
a property of arbitrary candidates, name the result in set-system-, matrix-,
graph-, or polynomial-neutral terms rather than attaching theorem-only
semantics.

Serialization does not preserve trusted provenance. Every value reconstructed
from a public payload is caller-authored, even when its bytes are identical to
the output of a recognizer. A theorem-dependent consumer must therefore rely on
one of three enforceable boundaries: ordinary construction of the canonical
subtype validates the theorem property, the value carries source-bound evidence
checked by an explicit bounded verifier, or the consumer performs recognition
itself. A subtype name, `validated` boolean, digest, or claim that a producer ran
earlier is not evidence. Test native composition and serialized public
composition separately when they cross different trust boundaries.

Likewise, an exact-success result must make failure of its defining invariant
unrepresentable. When result branches change the presence or mathematical
meaning of a witness, certificate, diagnostic, or derived value, use a
discriminated result such as `CONSTRUCTED`, `NOT_APPLICABLE`, or `UNKNOWN`.
The generated public schema must expose those branches and exclude their
contradictory field combinations; do not replace that contract with one model
containing optional fields and corrective booleans. If all branches genuinely
share one field shape, cheap structural validation must still enforce every
status implication. A result must not report exact success while also reporting
that reconstruction, target matching, optimality, or certificate validation
failed. Diagnostic fields may explain a non-success branch; they cannot weaken
the operation's advertised postcondition.

This rule does not require ordinary result construction to replay an expensive
proof. The trusted kernel must establish the defining invariant before calling
its private construction path, while owning tests replay the invariant on
known-answer, adversarial, and property-based fixtures. Independently supplied
claims still use the explicit bounded verifier described below.

These checks have distinct owners. Catalog admission decides whether an
operation is published; it does not admit a particular runtime request.
Request admission proves that one parsed request belongs to the advertised
mathematical and execution envelope and produces the owner-local execution
plan. Backend conversion converts an already-admitted value; it does not widen
or discover that domain. Result construction converts the output of that
execution into the canonical typed result. The defining invariant is
primarily established by the operation's known-answer, defining-identity, and
adversarial tests; ordinary execution must not replay the entire mathematical
computation just to construct its own result. Adapters may reject malformed
backend data during conversion, but that is integration safety rather than a
separate mathematical result stage.

When result construction needs to bypass semantic replay, expose one private
owner-local factory such as ``_from_kernel``. It may use trusted construction
only after the kernel has established every invariant it skips. Pydantic result
validators remain limited to structural, linearly bounded checks; they do not
call a backend, enumerate a search space, invoke a solver, or recompute the
operation's defining relation. An independently supplied claim instead uses
the explicit verifier below.

Do not use ordinary result construction to validate independently supplied
results. When a request or consumer can provide result data for verification,
use an explicit replay verifier with a declared replay-work ceiling and a
soundness statement for the claim being checked. Transport limits are not a
substitute for that replay bound, and a cheap identity must not authenticate a
stronger claim such as canonicity, irreducibility, or non-existence.
When that verifier is a public ``-> bool`` predicate, structurally valid but
out-of-envelope or admission-rejected claims return ``False`` rather than
leaking a validation exception. This fail-closed result is about the supplied
claim; unexpected owner or backend faults remain operational failures and must
not be converted into a mathematical negative.

Public numeric values are canonical exact rationals. IEEE doubles may exist
only inside a private kernel; any double crossing the boundary is carried as
its exact dyadic rational (`Fraction(float(v))` is lossless), because the
transport rejects JSON floating points and results must stay reconstructible.
Numerical backends such as Golub-Welsch therefore compute in floats but return
dyadic-exact values with their admission bounded to the finite-double range.
This invariant is enforced by the canonical transport and the integration
examples lane; meet it at design time rather than relying on those tripwires.

Do not add a public operation until its stated mathematical claim has a bounded,
appropriate implementation. A public operation is the `MathTool` contract—its
identifier, typed request and result, scope, and mathematical claim—not merely
a native Jacobian function or maintained backend routine. It may adapt either,
but its claim must be no broader than the implementation can establish.

A heuristic or approximation may be useful only when its result contract states
that limited scope. It must not return a negative decision, exact invariant, or
optimum that the implementation cannot establish.

Verify that the adapter preserves the claimed semantics. Do not present a
heuristic, approximation, or solver `UNKNOWN` as an exact conclusion; coerce
exact values to floating point; confuse similarly named invariants; or discard
backend information the result contract needs, such as multiplicities, bases,
or witnesses.

Before declaring the operation, provide tests for:

- a known-answer input and its claimed mathematical result;
- a boundary or degenerate input, including valid empty, zero, singleton, or
  identity values where the domain admits them;
- an adversarial input that distinguishes the stated semantics from a tempting
  weaker algorithm;
- a public-operation assertion that the returned value satisfies its defining
  mathematical invariant or witness, rather than merely parsing or reaching a
  backend; and
- request validation proving a schema-valid input either returns a typed
  result or is rejected by the request model—never a host exception.

Apply these adapter and request-boundary rules:

- Every public string field that carries mathematical syntax must document a
  finite grammar and have a test proving that parsing does not evaluate caller
  text. Do not pass caller input to `sympify`, `parse_expr`, `eval`, `exec`, or
  an evaluator generated by `lambdify`. Prefer canonical term or AST values as
  the authoritative contract; a textual convenience parser, if one exists,
  must construct the same value from an explicit allowlist.
- For every backend routine, record the coefficient domain, dimensional or
  degree limits, structural preconditions, degenerate cases, and resource
  limits it accepts. Encode those constraints in the concrete request model so
  an accepted request does not discover the backend domain through an
  exception.
- Every exact decomposition, certificate, or authoritative derived value must
  state its defining reconstruction or preservation equation and test it. Do
  not infer a mathematical property from the shape of lossy backend output or
  discard units, multiplicities, generators, axes, quotient maps, or other data
  needed to reconstruct or compose the result.
- Canonical integer and rational strings must reach backends only through
  `parse_canonical_integer()`, `as_integer_ratio()`, or an owner conversion
  helper. When the contract permits values above CPython's 4,300-digit integer
  string conversion limit, every adapter must include a test above that limit.
- For operations with mathematical preconditions such as nonsingularity,
  uniqueness, irreducibility, or nondegeneracy, tests must cover each excluded
  class and prove rejection occurs during request validation.

Before publication, record one owner-local admission decision in the
mathematical domain's `_admission.py` module. `jacobian.catalog.admission` owns
the shared policy types and fail-closed validation (see the
[public operation admission](public-operation-admission.md) contract).

### Domains, parents, and coercion

Canonical values carry the context needed to determine their mathematical
meaning. A polynomial includes its coefficient domain, generators, and
ordering where relevant. An ideal belongs to exactly one polynomial ring. A
finite-field element includes its field presentation. Matrices and
authoritative derived tables retain their axes and parent domain.

The same serialized expression in two contexts need not denote the same value.
Require exact parent identity for ordinary operations. A deliberate change of
ring, field, parent, generators, or axes must use a named operation or typed
morphism whose behavior is part of its contract. Never silently map unmatched
variables to zero. Backend generator inference, ambient rings, and automatic
coercion are private conveniences and do not define Jacobian's public
semantics.

### Canonical-value ownership check

Before adding a mathematical value, search the existing `values.py`,
`_models.py`, and native exports by semantic meaning and fields, not only by
class name. Reuse the existing owner across requests, results, producers, and
consumers whenever possible. If a distinct type is necessary, record the
different parent, representation, or postcondition, and define an explicit
typed conversion rather than relying on caller-side reconstruction.

Each mathematical value has one canonical type owned by its domain. A producer
returns that type and downstream consumers accept it unchanged. An operation
request may contain the value alongside genuine operation parameters, but must
not redefine the value as a parallel collection of fields. Callers must not
have to remember and reattach a field, ordered axis, ranked signature, ambient
dimension, or other mathematical context. This closure rule applies to empty
and degenerate values too: for example, a zero-row matrix still retains its
declared column axis.

When a producer-consumer relationship exists, the operation review artifact
must name it and its tests must pass the producer's serialized value directly
through the consumer's typed boundary. Do not introduce a generic value
registry or universal mathematical-object base class for this purpose; reuse
the owner domain's concrete value type.

Classify public outputs before choosing their schema:

| Output kind | Contract |
| --- | --- |
| Canonical value | A complete reusable mathematical object accepted by its downstream consumers. |
| Source-bound result | A source value plus a conclusion or certificate whose defining relation is validated. |
| Display projection | A human-readable summary that is not accepted as a composable mathematical value. |

For every producer or materially changed consumer, answer all of the following
in the producer/consumer closure field of the review artifact:

- What domain-owned canonical type does the producer return?
- Which downstream operations consume that type?
- Can its serialized value be supplied to each consumer unchanged?
- Does it retain its parent, presentation, ordered axes, ambient dimension, and
  normalization where those determine its meaning?
- What mathematical context remains present for empty, zero, identity, or
  otherwise degenerate values?
- Is each decision or certificate bound to the source value it concerns?
- If independently supplied result data is accepted, can its explicit verifier
  replay the defining relation within a declared work bound?

Decision and profile results are relations, not detached booleans or numbers.
Retain the source values needed to state the relation. When the public contract
accepts an authored conclusion or certificate, replay its defining equation in
an explicit bounded verifier. A compact result may omit a large derivation
ledger when bounded replay from the retained source is deterministic, but it
must not accept an authored conclusion merely because its scalar fields have
the right shape.

Backend integration follows the reusable
[mathematical backend contract](mathematical-backends.md).

### Static and executable enforcement

Keep static policy limited to boundaries that syntax can identify reliably.
The architecture checker forbids evaluator-capable parsing in the mathematical
tree and confines process execution to explicit owners behind the shared
supervisor. The fail-closed admission ledger and catalog conformance tests—not
an approximation in the linter—prove that public declarations use the standard
validation and execution path and do not expose backend values.

Mathematical correctness, parent compatibility, reconstruction, and backend
domain support require executable contract and property tests. Do not encode
those semantic claims as source-text or private-helper lint rules.

### Boundedness proof

Jacobian's operations are reusable mathematical instruments for agents doing
high-level mathematics and investigating conjectures. Treat boundedness as
part of the mathematical contract, not as a property discovered only by the
transport or a final serializer. Separate four obligations and assign them to
their owners:

1. **Semantic domain:** what stable mathematical map, predicate, invariant, or
   construction does the operation represent, independently of one release's
   execution limits?
2. **Admitted execution envelope:** which representations, mathematical
   objects, and degenerate cases may one request contain, and which controlling
   quantities bound that finite region?
3. **Computation:** what bounds the algorithm's work and intermediate values
   before the backend expands, enumerates, or solves anything?
4. **Output:** what bounds the exact returned value, witness, residual, or
   certificate, and how is that bound related to the admitted request?

The operation's mathematical contract owns the semantic domain and result
meaning. The domain owner owns request admission and the execution plan. The
kernel or maintained backend performs only the work described by that plan.
Result construction owns conversion; the operation's tests own defining-
invariant evidence. Dispatch and MCP own only the final transport projection.
A transport failure must therefore
be rejected by request admission or returned as a typed operational outcome;
it must not first appear as an uncaught post-execution serializer exception.

The operation identifier and result semantics own the first obligation. The
request contract enforces the second and the preconditions needed for the
third and fourth. Tightening or widening a safe execution envelope must not
silently change the mathematical meaning of the operation.

A backend or result conversion may still validate an invariant, but it must
not be the first place an accepted request discovers that its exact answer is
too large. If a bound is conservative, name the quantity it bounds, state why
it is safe for the algorithm, and test both the rejected adversarial case and a
useful case near the boundary. Do not use a post-hoc output-term cap,
truncation, sentinel, or host exception as a hidden computational budget.

For every non-trivially priced operation, owner tests must instrument the
priced kernel primitives on a representative near-envelope request and assert
that executed units do not exceed the admission charge. Pair that parity proof
with an adversarial useful request whose actual work fits, so a stale or
over-broad estimate cannot survive merely by rejecting it. Use the test-only
``tests.fixtures.accounting.assert_charged_work_parity`` helper; include every
instrumented primitive in the owner-local mapping, and do not add a shared
production ledger.

### Execution time and deadline composition

Exact mathematical work may legitimately take minutes or longer. Unless an
explicit service contract requires otherwise, do not use one short wall-clock
limit for every operation. Derive the admitted work from mathematical and
representation-specific quantities, then choose a killable wall limit that is
large enough for useful admitted requests. Wall time is a safety backstop, not
the proof that the computation is bounded.

Do not shrink an exact mathematical domain merely because a realistic admitted
computation is slow. First derive its work, intermediate, memory, and output
bounds; then improve the algorithm or backend and provide a generous killable
deadline. When no latency service level applies, calibrate that deadline from
realistic admitted workloads with explicit margin rather than inheriting a
universal short default. If callers may select wall time, expose only an
operation-owned range whose upper bound remains compatible with the admitted
work, memory, and cleanup envelope.

One accepted request has one owner-local execution plan. Parsing, normalization,
presolve, backend calls, exact replay, result validation, serialization, and
cancellation cleanup consume its shared deadline and charged work quantities;
no phase receives a fresh hidden budget. This is an owner-level contract, not a
generic production ledger. A caller or MCP read timeout must cover the declared
operation envelope plus bounded transport overhead; a shorter outer timeout may
abort the call, but it cannot establish an operation result.

For a killable subprocess or interactive backend, that envelope begins before
input spooling, launch, resource setup, and reader/writer startup. It also
covers termination, pipe draining, and reaping: reserve a named finite cleanup
allowance from the admitted deadline rather than granting a fresh post-timeout
clock to shutdown.

Preflight raw shape, nesting depth, digit length, and every derivable source
limit before copying or canonicalizing containers, constructing nested models,
parsing unbounded integers, or reaching a backend. A semantic bound may retain
useful cancellation cases, but its cheap source-side consequence must be
checked before expensive conversion.

When execution does not complete, retain the operation ID and version, exact
request or canonical digest, declared budgets, elapsed time, typed result or
outer error, timeout layer, backend status, and repository revision. Retry only
with an explicitly changed budget, backend, representation, or deterministic
partition, and retain both attempts.

### Representation-sensitive expansion

Representation is part of the execution envelope, even when two encodings
denote the same kind of mathematical object. Complete the following review
before selecting the kernel or request bounds:

- Is the input materialized, succinct, generated, or oracle-backed?
- What expansion does the kernel perform?
- Can admission bound that expansion before execution?
- Does an apparently equivalent compact representation change the complexity
  class or output obligation?

Name the accepted representation in the public contract and derive every
expansion budget from its canonical fields. Do not admit a compact value and
discover only inside the backend that it expands into too many states, terms,
assignments, or support points. Generated and oracle-backed inputs need the
same finite, deterministic source and work contract as materialized inputs; if
that contract cannot be stated and validated before execution, narrow or
reject the representation.

For example, exact total variation between two materialized finite tables is a
linear pass over their aligned support. Accepting succinct product
distributions is not merely a wire-format convenience: expanding their joint
support can be exponential, and computing the same invariant from the compact
representation can be a materially harder problem. Those representations
therefore need separate admission evidence and may require different
operations or result semantics even though the mathematical formula is the
same.

### Choose the controlling quantity

Use the quantity that actually controls work or output. A fixed ceiling on a
convenient field is appropriate only when its derivation conservatively bounds
the relevant computation and result. Prefer result-sensitive or
algorithm-sensitive admission when it preserves substantially more of the
useful mathematical domain.

For example, exact `binomial(n, k)` is output-sensitive: a middle binomial
coefficient may have an enormous decimal expansion, while `binomial(n, 0)` and
`binomial(n, 1)` remain compact for large `n`. A predicted result-digit bound
is more faithful than a uniform small ceiling on `n` when the kernel can avoid
constructing an over-budget result.

By contrast, factorial or binomial valuations have logarithmic digitwise
formulas and compact results even for very large arguments. Their useful
envelope should be derived from canonical input digit length, division or
base-digit steps, intermediate growth, and result digits—not inherited from
the much smaller region where materializing the factorial or binomial is
practical. If a desired value is a cheap composition of existing operations,
prefer widening the controlling primitive over publishing a near-duplicate
operation solely to escape an arbitrary cap.

Use measurements on representative and adversarial boundary fixtures to
choose the largest useful conservative envelope supported by the maintained
kernel. Measurements show usefulness; the mathematical work and growth
analysis remains the safety proof. Name each enforced budget in code and make
rejections identify the controlling quantity that was exceeded.

### Finite enumeration budgets

Large finite enumeration is compatible with a bounded exact operation. Admit it
using the mathematical quantities that actually control the computation rather
than treating small inputs as a goal in themselves. Record independent bounds
for:

- the number of candidate objects inspected in the worst case;
- the maximum intermediate height, degree, term count, or other value growth
  within one candidate computation; and
- the maximum number and canonical serialized size of returned values or
  witnesses.

Do not compress these into a convenient input product unless a documented
derivation proves that the product conservatively bounds every relevant
quantity. For example, a planar search over triples and quadruples has

```text
candidate_count = C(n, 3) + C(n, 4)
```

while coordinate height controls determinant intermediates and the result
shape controls witness serialization. `n * coordinate_digits` does not by
itself state any of those obligations.

A decision or first-witness operation must admit the full negative-case work,
but may have constant-size output. An all-witness operation or complete profile
must additionally admit its worst-case witness count and serialized result.
Prefer retaining the source value once and referring to indexed components over
repeating labels, parents, or other source context in every output entry.

Use measurements on representative and adversarial boundary fixtures to choose
useful conservative ceilings, but do not present timing measurements as the
boundedness proof. The proof is the finite candidate count and intermediate and
output growth; measurements establish whether the admitted region is useful on
the supported implementation.

When the complete search exceeds a single-call ceiling, deterministic
partitioning is acceptable only when the partition is a stable mathematical
subdomain and the result identifies exactly what was searched. The caller may
compose disjoint partitions, but no partition may claim global absence or
completeness. A timeout, node limit, or truncated witness list is not such a
claim. Keep the transport envelope separate from the mathematical output bound,
but make the relationship executable: the mathematical result bound must imply
that the complete canonical serialized result—including envelope metadata,
echoed fields, IPC framing, and JSON overhead—fits the transport limit. The
operation's estimator or exact size calculation must be shared by request
admission and the final projection, and boundary tests must exercise the
serialize -> parse -> canonicalize round trip. An accepted request must not
discover transport overflow only after the backend has run.

A transport limit is a concrete ingress, egress, IPC, persistence, or host
limit—not a speculative JSON-size policy. Do not truncate exact mathematical
values or add synthetic aggregate response budgets. Diagnostics must redact raw
caller values and bound their structural fields. Discovery remains exact cursor
pagination; when needed, bound immutable catalog metadata at declaration time
rather than trimming a page after projection.

When an operation has a genuine incomplete or unknown outcome, expose that
state in its domain result with the evidence and bounds needed to interpret it.
Do not turn an inability to finish or represent the exact answer into a
mathematical conclusion. When no such result is defined, narrow the request
domain until every accepted request returns the declared typed value.

CI executes every advertised invocation example and a bounded deterministic
mutation set derived from those examples. For every mutation accepted by the
concrete request model, the operation must return its declared result type and
must not leak a host or backend exception. The adversarial semantic case and
the schema-valid request-boundary case still belong in the owning domain tests;
generic mutations can expose admission gaps but cannot prove domain-specific
mathematical correctness.

If no bounded implementation can support the public claim, do not expose the
operation yet. A backend import or native function is not evidence that its
result has the desired mathematical semantics.
