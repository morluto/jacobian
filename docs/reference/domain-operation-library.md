# Domain operation library

Every built-in operation is a direct typed mathematical function with one
domain owner. Declaration modules export immutable tuples of
`MathTool` values. `math.find` matches those entries or returns one exact
contract, and `math.run` validates then executes exactly one operation.

The canonical operation path and ownership boundaries are defined in the
[architecture](../explanation/architecture.md). A domain function may use a
maintained library privately for its algorithm;
callers see Jacobian's typed mathematical values, not backend objects. When an
execution plan is needed, it is request-scoped and internal. It is not
caller-owned workflow state and does not add another MCP operation.

Keep values, codecs, invariants, and backend conversions with their domain.
Shared contracts are limited to passive cross-domain primitives. A bounded
operation reports mathematical completeness or uncertainty in its own result.

Public request contracts must make their valid representation visible before a
backend call. Express constraints that JSON Schema can represent in typed field
metadata. When a structural cross-field relation or canonical ordering needs a
Pydantic model validator, also provide an explicit field or model description
and a minimal valid example in the exported schema. The Pydantic model is
authoritative for the serialized request contract; the metadata lets a caller
form a valid first request rather than discover the rule only through a
rejected call. Schema and model validation express structural representation
constraints only: field types and bounds, canonical encoding, container shape,
and cheap intrinsic cross-field consistency. They must not invoke a backend,
enumerate candidates, select a kernel, reserve result work, or retain an
execution plan in a Pydantic private attribute. The owner admission function
runs after parsing and is shared by native and MCP execution.

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

Keep canonical carriers generic across owners when they represent the same
mathematical value. If a limit belongs only to one operation's admitted
execution envelope, enforce it in that operation's admission path and expose
it in its request schema or description. Do not create a bespoke value subtype
or schema override merely to move an operation-specific ceiling into a shared
carrier. A new subtype is appropriate when it changes the mathematical meaning
or establishes invariants that all of its consumers need.

The logic family illustrates the boundary. `sat.cnf.canonicalize` returns a
canonical CNF value; `sat.assignment.check` and `sat.solve` accept that value
directly. `smt.solve` accepts one bounded QF SMT-LIB query.

## Codomain closure

Start with the [default value contract](value-interoperability.md#default-return-the-value).
Correcting an incomplete value representation does not call for a certificate
wrapper. Return the mathematical value with its required interpretation.

Before admitting an exact operation, verify that canonical public value types
can represent every value in its advertised codomain. Each returned value must
have a backend-independent identity, an explicit mathematical parent and
interpretation when those affect its meaning, and enough information for exact
reconstruction and producer-consumer composition. Mathematically distinct
values must remain distinguishable after serialization.

Check expanded dimensions against both the result carrier and downstream
consumers. A shared label or axis limit must not silently impose a smaller
capacity than the matrix it describes. Reuse an existing carrier's supported
capacity when appropriate, while retaining the bounds of containers that own
more expensive work. Before adding an expansion guard, check whether existing
source admission already proves it. Do not duplicate that check or bypass
public round-trip validation with trusted construction.

A backend expression or ambient symbol is not a public mathematical identity.
Values that depend on a field, ring, module, coordinate system, embedding,
analytic branch, orientation, or basis must carry that context or bind to a
canonical domain-owned presentation of it. Evidence such as an isolating
interval, isolating rectangle, certificate, or decomposition may establish a
value's identity, but incidental choices of evidence must not create different
canonical values.

When the current value vocabulary cannot represent the complete codomain, add
the missing domain-owned carrier first. Do not narrow an advertised complete
result to representable cases, expose backend-relative expressions, collapse
conjugate or branch-dependent values, or report completeness after omitting
unrepresentable results. Intentional changes of parent or interpretation remain
explicit typed maps.

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
- Codomain closure and required parent, embedding, branch, or coordinate data:
- Canonical public value type:
- Result type:
- Reconstruction or defining invariant:
- Request representation, admission quantities, and degenerate cases:
- Result-size bound and exact-success/non-success states:
- Evidence: defining invariant plus the changed boundary regression:

### Worked boundary: a complete bounded profile

Consider an operation that receives a finite graph `G` and a non-negative
deletion order `b`, and returns the chromatic number of every graph obtained by
deleting at most `b` source edges.  Its semantic postcondition is the canonical
source-indexed map

\[
F \longmapsto \chi(G-F) \quad \text{for every } F\subseteq E(G),\ |F|\leq b.
\]

This is one complete profile. A sound contract makes the following choices
explicit:

- **Canonical rows:** order each source-edge subset by increasing cardinality
  and then lexicographically by source-edge index, including the empty subset.
  Every eligible `F` appears exactly once.
- **Result semantics:** each row carries the exact chromatic-number result for
  its own `G-F`.  The outer profile is `COMPLETE_EXACT` only if every row is
  exact; rows that are unknown or incomplete are retained explicitly rather
  than omitted or silently treated as a negative conclusion.
- **Admission:** with `m = |E(G)|`, the request preflights
  \(\sum_{k=0}^{b}\binom{m}{k}\) rows, the predicted work of each row, and
  the full serialized profile.  All row work shares one deadline and ledger.
- **Defining tests:** `b=0`, edgeless graphs, source-edge permutation, and
  small exact profiles such as `K_4` with `b=1` (one row of value 4 and four
  rows of value 3) distinguish source indexing, completeness, and row
  semantics.
- **Non-goals:** selecting a deletion set, minimizing over an unbounded graph
  family, recognizing a theorem-specific criticality class, and deriving a
  Ramsey or extremal conclusion remain caller composition.

The exhaustive source relation is the defining invariant. A graph-colouring
backend computes individual rows; choosing deletions or drawing a larger
conclusion remains caller composition.

### Validated mathematical subtypes and exact-success states

A structurally valid candidate is not a theorem-bearing value. A raw set
system, matrix, graph, polynomial, or similar carrier establishes only its
canonical representation and cheap intrinsic consistency. A meaningful
structural refinement, such as a monic polynomial, can preserve the shared
encoding while checking that intrinsic constraint.

An operation that depends on a stronger mathematical property—such as greedoid
axioms, antimatroid union closure, irreducibility, or independence of a proposed
basis—must establish that property under its own admitted contract. It may
consume a proposed witness or perform recognition and return a typed negative
or non-applicable outcome. A positive recognizer retains the source and the
mathematical conclusion so consumers can accept its value unchanged. A later
consumer checks that conclusion only when it relies on the authored claim;
ordinary use of the value does not require verifying its history.

Do not invoke mathematical backends in a public value constructor to make its
name theorem-bearing. Reuse domain-owned recognition helpers at the admitted
boundary. Within one trusted admitted execution, reuse established facts when
constructing the result instead of proving them again. If an operation
intentionally computes a property of arbitrary candidates, name that scope
rather than attaching theorem-only semantics.

Serialization does not preserve trusted provenance. Every value reconstructed
from a public payload is caller-authored, even when its bytes are identical to
an earlier output. A consumer that depends on a mathematical property must
establish that property as part of its own admitted operation contract. A
subtype name, `validated` boolean, digest, or claim that a producer ran earlier
is not evidence. Test native composition and serialized public composition
separately when they cross different trust boundaries.

Likewise, a computed exact-success result must satisfy its defining invariant
and exclude structurally contradictory status and witness combinations. When
result branches change the presence or mathematical meaning of a witness,
certificate, diagnostic, or derived value, use a
discriminated result with the branches actually defined by the operation, such
as `CONSTRUCTED` and `NOT_APPLICABLE`.
The generated public schema must expose those branches and exclude their
contradictory field combinations; do not replace that contract with one model
containing optional fields and corrective booleans. If all branches genuinely
share one field shape, cheap structural validation must still enforce every
status implication. A result must not report exact success while also reporting
that reconstruction, target matching, optimality, or certificate validation
failed. Diagnostic fields may explain a non-success branch; they cannot weaken
the operation's advertised postcondition.

Advertise `UNKNOWN` only when it is an explicit, representable result branch
with defined semantics. A backend's internal uncertainty does not add that
branch to an exact operation. In particular, timeout, unavailable runtime, or
exhausted execution resources must not become a factorization, divisor list,
negative answer, or mathematical non-applicability conclusion.

Distinguish invalid input from an established mathematical rejection. If an
operation asks whether a quotient construction applies, a well-formed relation
that fails the required compatibility condition is a typed negative outcome
when that outcome belongs to the declared codomain. Malformed encodings and
requests outside the admitted domain remain input/admission errors; unexpected
backend faults remain execution errors. Keep descriptions, result schemas,
native behavior, and MCP projection consistent with that distinction.

Ordinary result construction must not replay the computation that produced it.
The trusted kernel establishes the defining invariant before calling its
private construction path, while owning tests replay the invariant on
known-answer, adversarial, and property-based fixtures. Caller-supplied claims
are handled by the specific operation that consumes or checks them; they do not
require a general result-replay layer.

Validation and verification are separate responsibilities:

| Boundary | Permitted work |
| --- | --- |
| Request parsing | Canonical shape, grammar, nesting, digit, and raw representation bounds |
| Worker-output decoding | Strict codec, canonical scalar syntax, cardinality, and projection shape |
| Result structural validation | Axis alignment, references, coverage, and discriminated-state consistency |
| Trusted result construction | Owner-local `_from_kernel` construction after the kernel established the skipped invariants |
| Ordinary result deserialization | Canonical structural parsing only; it does not authenticate mathematical truth |
| Claim-checking operation or tests | Owner-specific mathematical work under that operation's admission |

Claim verifiers check the smallest relation a consumer needs from the retained
source and witness. They should not generally rerun the producer. For example,
a Pratt certificate retains the prime-power factorization of `p - 1`, recursive
subcertificates, and a modular witness; its verifier reconstructs that supplied
factorization and checks the witness equations without factoring `p - 1`.

No ordinary boundary may factor, isolate roots, enumerate candidates, invoke a
solver or backend, recompute a defining relation, or trigger a nested public
validator that performs that work. A computed result is a trusted producer
output. Public deserialization establishes its canonical representation, not a
second proof of its mathematical postcondition.

Public results describe mathematical meaning, not the implementation used to
compute it. Do not expose a constant backend, method, algorithm, exactness,
determinism, or verification field merely to narrate trusted execution. Retain
a field only when its value can vary and changes how callers must interpret the
mathematical result. Examples include exact versus unknown status, complete
versus bounded coverage, a normalization convention, approximation error, or a
caller-selected algorithm that is part of the public contract.

These checks have distinct owners. Catalog admission decides whether an
operation is published; it does not admit a particular runtime request.
Request admission proves that one parsed request belongs to the advertised
mathematical and execution envelope. It may also produce an owner-local
execution plan when the kernel needs reusable derived facts. Backend conversion
converts an already-admitted value; it does not widen
or discover that domain. Result construction converts the output of that
execution into the canonical typed result. The kernel establishes the returned
instance's mathematical postcondition; known-answer, defining-identity, and
adversarial tests independently check that implementation. Ordinary execution
must not replay the entire mathematical computation just to construct its own
result. Establishing a backend candidate's certificate conditions belongs to
the producing kernel, as specified in the
[backend contract](mathematical-backends.md#common-adapter-obligations).
Adapters may reject malformed backend data during conversion, but that is
integration safety rather than a separate mathematical result stage.

Request-model validation is not a second execution-plan layer. A raw
``mode="before"`` preflight may reject cheap representation facts—such as
shape, nesting, digit length, or an aggregate source limit—before expensive
canonicalization. It must not perform full candidate enumeration, invoke a
solver or backend, or replay the operation's defining relation merely to
validate a request. After canonicalization, the owner makes one semantic
admission decision for the invocation. It reuses any derived facts through the
kernel and trusted result construction; request validators and operation
wrappers must not recompute the decision independently. A simple admission
guard may return nothing on success. Do not introduce a plan class or module
unless it carries information that a later phase genuinely consumes.

Result construction uses one private owner-local factory such as
``_from_kernel`` whenever public validation would replay semantic work. It may
use trusted construction
only after the kernel has established every invariant it skips. Pydantic result
validators remain limited to structural, linearly bounded checks; they do not
call a backend, enumerate a search space, invoke a solver, or recompute the
operation's defining relation.

When removing repeated validation, trace each removed invariant to its owner.
Removing replay does not authorize removing recognition of a caller-supplied
claim. Review every public entry point, including convenience constructors and
native functions: an unchecked public factory must not bypass an authenticated
operation. Keep trusted factories private rather than introducing compatibility
paths that preserve the bypass.

Do not use ordinary result construction to validate independently supplied
results, and do not add a companion checker by default. A compute-only
operation normally needs defining-invariant tests, not production replay. If
checking a caller-authored claim is independently useful, expose a semantically
specific operation whose accepted domain, work bound, and postcondition state
exactly what it checks. Transport limits are not a substitute for mathematical
admission, and a cheap identity must not authenticate a stronger claim such as
canonicity, irreducibility, or non-existence. Unexpected owner or backend faults
remain operational failures and must not be converted into a mathematical
negative.

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
- owner-boundary evidence distinguishing a schema-valid mathematical rejection
  from typed operational non-completion. Raw backend or host exceptions must
  never escape, and operational non-completion must never become a mathematical
  result.

Apply these adapter and request-boundary rules:

- Every public string field that carries mathematical syntax must document a
  finite grammar and have a test proving that parsing does not evaluate caller
  text. Do not pass caller input to `sympify`, `parse_expr`, `eval`, `exec`, or
  an evaluator generated by `lambdify`. Prefer canonical term or AST values as
  the authoritative contract; a textual convenience parser, if one exists,
  must construct the same value from an explicit allowlist.
- For every backend routine, record the coefficient domain, dimensional or
  degree limits, structural preconditions, and degenerate cases it accepts.
  Enforce structural constraints in the request model and mathematical
  admission in the owning domain before backend invocation. Native and MCP
  callers use that same admission path once; an admitted request must not
  discover an unsupported backend domain through an exception.
  Keep configured worker and host capacity limits in the adapter or deployment;
  exhaustion there is an operational failure, not invalid input.
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
  class and prove the owner rejects it before backend execution.

Before publication, add the declaration to the mathematical owner's `_tools.py`
manifest. Presence there is the publication decision; native-only functions
remain ordinary package exports without a declaration. Catalog construction
fails closed on malformed manifests and duplicate operation IDs (see the
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

Return ordinary exact values directly. Certificates are not a universal result
requirement: include a witness when it is requested, supports construction, or
has a distinct independent-checking purpose. See
[when a witness or certificate is useful](value-interoperability.md#values-witnesses-and-source-binding).
A consumer of a value need not verify its producer's history; it checks an
authored relation only when its own result relies on that relation.

See [schemas, mathematical values, and conversions](value-interoperability.md)
for the representation decision table, schema/parser agreement, conversion
publication rules, and serialized trust boundary.

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

When a producer-consumer relationship exists, the issue, PR description, or
requested audit must name it and its tests must pass the producer's serialized value directly
through the consumer's typed boundary. Do not introduce a generic value
registry or universal mathematical-object base class for this purpose; reuse
the owner domain's concrete value type.

### Contract-audit red flags

Review a changed public request, result, or value against these patterns before
adding another field or validator. They are evidence to inspect the owning
contract, not automatic rules for adding a new public operation.

| Observed pattern | Required review |
| --- | --- |
| A result returns a raw nested tuple, list, or dictionary representing a matrix, vector family, map, partition, or indexed relation | Find the domain-owned value. It must retain its ring or field, ordered axes, parent, and empty shape; otherwise add or repair that value rather than duplicating context in sibling fields. |
| A result has `rows`, `columns`, `dimension`, `rank`, `source`, or `axis` fields beside raw mathematical data | Determine whether those fields reconstruct context that belongs inside one canonical value. Keep only metadata that states the operation's distinct postcondition. |
| A model validator factors, proves primality, runs elimination, computes closure/orbits, checks independence, or evaluates a defining equation | Move the semantic work to named operation admission. Migrate every consumer that relies on it and preserve only bounded structural parsing. |
| A conversion changes a ring, field, basis, presentation, parent, or axis order | Name the source and target values and its applicability conditions in an explicit typed map. Keep extraction and formatting projections native unless they establish a reusable mathematical relation. |
| A subtype, digest, `verified` flag, or producer-shaped payload is treated as proof after transport | Treat it as a caller-authored claim. The consumer must admit and check the precise relation it uses; producer result construction must not repeat its completed computation. |

For each flagged case, add the smallest behavioral test that crosses the
affected public boundary: an empty or degenerate producer value where relevant,
a serialized producer-to-consumer round trip, and a forged structural claim
that reaches the consumer without the producer.

Classify public outputs before choosing their schema:

| Output kind | Contract |
| --- | --- |
| Canonical value | A complete reusable mathematical object accepted by its downstream consumers. |
| Source-bound result | A source value plus a conclusion or certificate whose defining relation the producer establishes. Publicly supplied claims require admitted consumer checking. |
| Display projection | A human-readable summary that is not accepted as a composable mathematical value. |

For every producer or materially changed consumer, record the applicable
closure evidence in the issue, PR description, or requested audit:

- What domain-owned canonical type does the producer return?
- Which downstream operations consume that type?
- Can its serialized value be supplied to each consumer unchanged?
- Does it retain its parent, presentation, ordered axes, ambient dimension, and
  normalization where those determine its meaning?
- What mathematical context remains present for empty, zero, identity, or
  otherwise degenerate values?
- Is each decision or certificate bound to the source value it concerns?
- If caller-supplied data is interpreted as a mathematical claim, which
  consumer operation establishes the property it needs?

Decision and profile results are relations, not detached booleans or numbers.
Retain the source values needed to state the relation. When a public contract
accepts an authored conclusion or certificate, its consumer must establish the
specific property it relies on. A compact result may omit a large derivation
ledger when the retained source and claimed postcondition remain sufficient for
that consumer, but scalar shape alone must not authenticate a mathematical
conclusion.

Backend integration follows the reusable
[mathematical backend contract](mathematical-backends.md).

### Static and executable enforcement

Keep static policy limited to boundaries that syntax can identify reliably.
The architecture checker forbids evaluator-capable parsing in the mathematical
tree and confines process execution to explicit owners behind the shared
supervisor. Owner-local tool manifests and catalog conformance tests prove that
public declarations use the standard validation and execution path and do not
expose backend values.

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
4. **Output:** what bounds the unavoidable cardinality and representation
   growth of the exact returned value, witness, residual, or certificate?

The operation's mathematical contract owns the semantic domain and result
meaning. The domain owner owns request admission and the execution plan. The
kernel or maintained backend performs only the work described by that plan.
Result construction owns conversion; the operation's tests own defining-
invariant evidence. Dispatch and MCP own only the final transport projection.
A transport-specific byte ceiling is not mathematical admission: when a real
delivery boundary configures one, exceeding it is a typed operational outcome.
Do not copy that ceiling into a request model, shared carrier, native function,
or owner work plan.

The operation identifier and result semantics own the first obligation. The
request contract enforces the second and the preconditions needed for the
third and fourth. Tightening or widening a safe execution envelope must not
silently change the mathematical meaning of the operation.

A backend or result conversion may still validate an invariant, but it must
not discover an unbounded cardinality or host representation failure only after
performing the work. If a bound is conservative, name the mathematical
quantity it bounds, state why it is safe for the algorithm, and test both the
rejected adversarial case and a useful case near the boundary. Do not use JSON
bytes, a post-hoc output-term cap, truncation, sentinel, or host exception as a
hidden computational budget.

Use encoded byte counts only at a concrete byte boundary: bounded process
stdin/stdout, a digest whose definition includes encoded bytes, or an explicitly
configured transport limit. The canonical encoder measures without an output
ceiling by default; pass `CanonicalLimits` only when the caller owns such a
boundary. Shared dispatch is not such a boundary: it preserves strict JSON
semantics without imposing an HTTP, stdio, or in-process payload size. An
unexplained reserve added to the codec's default is not an output
proof. The architecture check rejects canonical output defaults and result-byte
policies in mathematical carriers and operation owners, while allowing limits
whose names identify a concrete worker or process channel. Express mathematical
safety through cardinality, component digits, depth, or another intrinsic
representation quantity.

A concrete channel or host may reject work that exceeds its configured
capacity. That failure does not redefine the operation's mathematical domain.
MCP surfaces it through the ordinary tool-error path, not as `INVALID_PARAMS`;
operations do not need a shared capacity exception or admission-time prediction
of every deployment's available memory.

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

One accepted request has one owner-local execution envelope. Parsing,
normalization, presolve, backend calls, result construction,
serialization, and cancellation cleanup consume its shared deadline and
charged work quantities; no phase receives a fresh hidden budget. This is an
owner-level contract, not a generic production ledger. A caller or MCP read
timeout must cover the declared operation envelope plus bounded transport
overhead; a shorter outer timeout may abort the call, but it cannot establish an
operation result.

When admission derives work, intermediate, memory, or exact-output reservations,
compute them once after canonicalization and pass or otherwise reuse them in an
owner-local plan. Do not make a request model, operation wrapper, and trusted
result constructor independently repeat the same admission probe. Operations
without reusable derived facts need no plan object. Caller-supplied claims are
handled by the admission and kernel of the operation that consumes them.

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

Scale first when an estimate rejects a motivating valid workload. Preserve
relationships between intermediates, reduce the problem exactly, or improve
the representation, algorithm, or backend before retaining the restriction.
A sound but grossly pessimistic estimate can still be a product defect. Follow
the [execution-envelope review](public-operation-admission.md#execution-envelope-review):
a cheaply executable rejected case must become an accepted regression, not a
new permanent rejection test. Report an unresolved scaling limitation as
unfinished work rather than claiming that narrowing admission fixed it.

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
must additionally admit its worst-case witness count and retained exact data.
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
claim. Keep the transport envelope separate from the mathematical output bound.
A native or in-process operation proves cardinality, digit growth, depth, and
allocation safety without assuming a delivery format. When a concrete adapter
adds a byte ceiling, that adapter measures the complete encoded envelope,
including metadata, echoed fields, framing, and escaping. Worker adapters do
the same for their own stdin and stdout channels. Do not move either byte
ceiling into the shared request model merely so every possible consumer shares
the most restrictive transport.

Compare the complete canonical encoding with the actual enforcing limit. Do not
subtract an unexplained safety reserve or add a second, duplicate size probe:
either can reject useful requests without being the real boundary, while still
missing bytes introduced by escaping, framing, echoed context, or final
serialization. A finite cleanup or reaping allowance is different: name it,
bound it, and include it in the operation's documented deadline envelope.

A transport limit is a concrete ingress, egress, IPC, persistence, or host
limit—not a speculative JSON-size policy. Do not truncate exact mathematical
values or add synthetic aggregate response budgets. Diagnostics must redact raw
caller values and bound their structural fields. Discovery remains exact cursor
pagination; when needed, bound immutable catalog metadata at declaration time
rather than trimming a page after projection.

When an operation has a genuine incomplete or unknown outcome, expose that
state in its domain result with the evidence and bounds needed to interpret it.
Do not turn an inability to finish or represent the exact answer into a
mathematical conclusion. Runtime timeout, cancellation, or capacity exhaustion
is an execution error rather than an `UNKNOWN` mathematical result.

CI executes every advertised invocation example and a bounded deterministic
mutation set derived from those examples. For every mutation accepted by the
concrete request model, the operation must return its declared result type or a
typed operational non-completion; it must not leak a raw host or backend
exception. The adversarial semantic case and
the schema-valid request-boundary case still belong in the owning domain tests;
generic mutations can expose admission gaps but cannot prove domain-specific
mathematical correctness.

If no bounded implementation can support the public claim, do not expose the
operation yet. A backend import or native function is not evidence that its
result has the desired mathematical semantics.
