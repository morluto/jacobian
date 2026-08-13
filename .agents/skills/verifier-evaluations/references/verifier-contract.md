# Verifier Contract Reference

Use this reference when designing or reviewing an independent verifier. It is
deliberately implementation-oriented; the parent skill supplies the workflow.

## Contract map

Record these values before changing code:

| Boundary | Questions |
| --- | --- |
| Visible schema | Which fields, types, bounds, cardinalities, and enums are promised to the agent? |
| Instructions | Are labels, sign conventions, scope, limitations, and evidence rules explicit? |
| Hidden input | What exact source revision, row, fixture, or environment is frozen? |
| Mathematical claim | What is the semantic equality relation? Which equivalent witnesses are valid? |
| Evidence | Which schema-valid paths and digests are allowed? What symlink and traversal rules apply? Is any task-specific limit explicitly public and enforced? |
| Assurance | What is the maximum allowed claim? Who, if anyone, can authorize `VERIFIED`? |
| Diagnostics | Which dimensions are reported independently, and what forces aggregate reward to zero? |

The schema, instruction, hidden input, and verifier form one contract. A field
declared in only one of them is a drift risk.

## Safe verifier shape

Prefer a small sequence of total predicates:

```text
load bounded regular submission
  → bind visible input to one bounded regular frozen input
  → validate envelope and exact types
  → validate shape, bounds, and cardinality
  → validate semantic mathematics
  → validate evidence binding and scope
  → calculate independent diagnostic dimensions
  → force aggregate reward to zero for invalid protocol or false assurance
  → write deterministic reward and diagnostics
```

Every stage must turn malformed input into a false predicate, not an uncaught
exception. Catch file, JSON, recursion, and resource errors at the boundary;
do not wrap the whole verifier in a broad exception that hides a real bug.
Perform the visible-input binding before parsing visible JSON. Use the frozen
verifier copy as the semantic source after binding succeeds; frozen files may
have task-specific names, so discover or declare the sole candidate rather
than assuming `/tests/input.json` exists.

## Type and semantic traps

- Use exact type checks where the contract distinguishes booleans, integers,
  and floats: `type(value) is int`, not `isinstance(value, int)`.
- Compare rational values by normalized mathematical value, not by numerator /
  denominator spelling. Preserve denominator and sign constraints separately.
- Validate all declared ranges and `maxItems` limits in the verifier, even when
  the public schema declares them. Hidden callers can bypass schema validation.
- Validate nested objects before reading a member such as `value["n"]` or
  constructing `set(value)`. Unhashable malformed members must receive zero,
  not a traceback.
- Treat JSON numbers as untrusted input: reject non-finite values and catch
  `ValueError`, `TypeError`, and `OverflowError` before integer conversion or
  arithmetic. Do not let `1e309` become a verifier crash.
- Accept alternate order, scaling, or equivalent witnesses only when they are
  mathematically valid and within the task's declared scope.

## Evidence and resource safety

- Check evidence is a list of the required cardinality before resolving or
  hashing descriptors.
- Require exact relative paths, reject `..`, absolute paths, symlink components,
  non-regular files, and targets outside the workspace.
- Bound the submission and visible/frozen input before `read_text`,
  `read_bytes`, or JSON parsing. Benchmark evidence has no arbitrary byte cap;
  do not add a hidden `max_bytes` rule that is absent from the public contract.
  Evidence must still be schema-valid, digest-bound, at the declared exact
  path, and inside the verifier workspace.
- Bind the evidence digest to the exact result or certificate being scored.
  A valid digest for unrelated content is not evidence for the claim.
- If evidence validity is reward-bearing, reject unrelated nonempty prose.
  Publish the mathematical explanation obligations and test acceptable
  paraphrases; do not substitute hidden marker syntax or keyword soup.
- Keep evidence parsing type-sensitive. A marker containing JSON must parse to
  the expected object, not merely be present. The schema and resolver must
  agree exactly: remove unsupported descriptor properties rather than silently
  ignoring them.

## Scope, prose, and assurance

- Make scope an exact or structured contract: frozen source identity, named
  domain, completeness, and limitations.
- Do not use unrelated negation to rescue an affirmative claim. If text must be
  interpreted, evaluate it by clause and test both affirmative and negative
  paraphrases.
- Do not require arbitrary explanation keywords. Require a bound structured
  result and a documented explanation obligation, or expose the required fields
  directly.
- Reject `VERIFIED` unless an independent, operator-authorized record binds the
  exact claim, candidate, scope, certificate, and checker identity.
- Do not let an assurance or scope failure erase mathematical correctness in
  diagnostics; do let it prevent full aggregate reward.

## Minimum adversarial matrix

| Family | Examples |
| --- | --- |
| Envelope | missing field, extra field, wrong task ID, wrong conclusion, false `VERIFIED` |
| Types | `true` for integer, `64.0` for integer, `1e309`, scalar instead of object, unhashable member |
| Bounds | negative or oversized index, too many entries, huge submission or visible input, deep JSON; large valid evidence remains accepted when otherwise valid |
| Semantics | wrong answer, equivalent unreduced rational, reordered/scaled witness |
| Evidence | empty list, duplicate descriptors, wrong path, wrong digest, unrelated result, symlink, traversal |
| Scope | wrong source revision, incomplete claim, empty or arbitrary limitations, affirmative forbidden claim |
| Runtime | missing file, permission error, malformed JSON and wrong-shaped JSON in the submission and visible input, timeout-shaped input; assert reward artifact exists |

At least one test in each family should mutate a canonical submission and then
rebind any evidence digest deliberately. Tests should assert observable reward
and diagnostic fields, not private helper names or copied source text.

## Evaluation integrity and handoff

Separate these claims:

- a verifier unit test passed;
- a selected task contract passed;
- the exact Oracle produced full applicable reward;
- the repository gate passed;
- a benchmark observation is reproducible;
- an evaluation supports a causal operation claim.

The first four do not imply the last two. Record task digest, source revision,
Harbor/runtime version, model and settings where applicable, exact command,
selected scope, result JSON, and any skipped or unavailable checks.

For parallel work, record each agent's worktree, owned paths, commit, pushed
branch, and validation. Integrate shared generated changes once, then run the
final gate on the integrated tree.
