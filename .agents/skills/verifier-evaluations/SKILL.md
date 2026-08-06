---
name: verifier-evaluations
description: Design, audit, and repair fail-closed mathematical verifiers and evaluation contracts, including agent-visible schemas, hidden evidence binding, assurance and scope semantics, diagnostic scoring, adversarial fixtures, Oracle validation, and evaluation handoffs. Use when a verifier can crash, accept malformed claims, reject equivalent witnesses, leak answers, collapse score dimensions, or when an evaluation needs reproducible integrity evidence.
---

# Verifier Evaluations

Design the published contract and the executable verifier together. Treat the
verifier as an independent checker of an observable claim, not as a parser for
the expected solution. For Harbor packaging, use this skill with
`harbor-benchmarks`: this skill owns verifier and evaluation reasoning, while
the Harbor skill owns dataset layout and repository commands.

## Workflow

1. Establish the contract boundary. Read the agent-visible schema and
   instructions, hidden expected material, task metadata, evidence format, and
   assurance ceiling. List what is intentionally fixed, what alternate witness
   forms are valid, and what the verifier must not reveal. Make the public
   protocol explicit: the agent should receive the required output shape and
   types, allowed enum values, assurance ceiling, evidence paths and digest
   rules, scope, completeness, and artifact names. Do not expose the expected
   solution, hidden verifier logic, private authorization records, or Oracle
   fixtures merely to explain the format.

   Use `tests/verifier_contract.json` as the sole task-local declaration of
   behavior consumed by generic verifier tests. Keep it versioned and validate
   it against a closed schema: boolean fields must be exact JSON booleans,
   unknown keys and unsupported versions fail closed, and missing metadata must
   not silently opt a task into exceptional behavior. Do not add parallel
   metadata files, fields on unrelated public-contract models, or global
   task-name registries for input-binding, scope, assurance, or diagnostic
   exceptions.

2. Trace every acceptance path:

   `input file → envelope → typed structure → semantic claim → evidence/scope → metrics → reward`

   For each edge, identify malformed input, resource exhaustion, path escape,
   false assurance, and exception paths. A verifier must produce a deterministic
   reward artifact for every submission, including malformed submissions.
   Bind the agent-visible input to the sole frozen verifier input before any
   task-specific JSON parse, key access, or computation. After equality is
   established, evaluate semantics from the frozen verifier copy rather than
   reparsing attacker-controlled workspace bytes.

3. Implement fail-closed checks. Validate exact types, cardinalities, ranges,
   and object shapes before indexing, iterating, hashing, or constructing sets.
   Reject booleans where integers are required, reject non-finite numeric input
   before conversion, and catch conversion overflow. Use semantic comparison
   for equivalent rational or normalized witnesses. Bound regular submissions
   and visible/frozen inputs before reading to prevent resource exhaustion;
   benchmark evidence has no arbitrary byte cap, but must still reject symlink
   and workspace escapes and bind exact schema-valid descriptors to paths and
   digests. Reject `VERIFIED` without independent authorization.

4. Keep diagnostic dimensions independent. Compute protocol compliance,
   mathematical correctness, evidence validity, scope accuracy, assurance
   calibration, and workspace input binding separately. Aggregate them only at
   the final reward boundary; invalid protocol or false certification may force
   aggregate reward to zero, but must not hide which other dimensions passed or
   make a correct mathematical result look like a mathematical failure. In
   particular, a tampered or replaced `input.json` must not zero the
   mathematical-correctness score; report it as a separate `input_binding`
   dimension and gate only aggregate reward on it.

5. Make prose requirements structural. Do not award credit because a response
   contains words such as “duplicate”, “line”, or “region”. Prefer typed result
   fields, exact evidence bindings, frozen limitation values, and clause-aware
   semantic checks. Removing an undocumented JSON marker does not justify
   accepting any nonempty file. Any reward-bearing prose obligation must be
   documented in the visible contract, reject unrelated text, and accept
   mathematically equivalent phrasing.

   Minimize the semantic obligations before implementing prose checks. Require
   only logically independent facts that the evidence must contribute. If the
   typed certificate plus one checked fact already entails a conclusion, do
   not also require a rhetorical sentence restating that conclusion. For
   example, a verifier that proves the submitted corrected condition fails
   must not additionally require the solver to say “therefore this does not
   refute the repair.” Keep such implications in verifier-owned mathematics,
   not in preferred wording.

   For streamed prose verifiers, preserve the local relationship between the
   claim, its scope, and any negation; independent lexical matches are not a
   sufficient semantic parser. Add regressions for scope-before-claim and
   claim-before-scope wording, affirmative constructions such as “not only”,
   and negations that apply to an unrelated object. Feed those fixtures across
   chunk boundaries that split the relevant phrases, so buffering behavior
   cannot change the meaning. Keep both positive equivalents and genuine
   contradiction cases in the matrix.

6. Build adversarial fixtures before trusting the Oracle. Cover the public
   contract itself, including a correct witness with the permitted assurance
   level and a correct witness with an unsupported assurance claim. Also cover
   the canonical witness, alternate valid witnesses, wrong mathematics,
   booleans and floats,
   unreduced rationals, missing and extra fields, empty and unhashable nested
   values, out-of-range values, malformed JSON and valid JSON of the wrong
   top-level shape for both submissions and agent-visible inputs, non-finite
   numeric values, oversized submissions or inputs, unhashable nested values,
   symlink and traversal evidence, wrong digests, duplicate evidence
   descriptors, missing limitations, false `VERIFIED`, and assertions that the
   verifier still emits `reward.json` without crashing. Include a large valid
   evidence artifact to prove that no undocumented byte cap is present.

   Include at least one terse numeric or structural explanation fixture built
   independently from the public contract. It must express the required facts
   without reusing the canonical answer's labels, rhetorical conclusions, or
   sentence fragments. Do not inspect hidden solution text solely to construct
   this fixture; the point is to prove semantic acceptance rather than encode a
   second preferred answer.

7. Validate the final tree and handoff. Run focused tests, deliberate negative
   cases, the selected Oracle, and the repository's planned gate. If shared
   verifier support changes, migrate only deliberately selected task-local
   copies, refresh only their checksum labels, and update only affected
   prospective adapter evidence; never rewrite historical snapshots.
   Report exact revisions, commands, checks, Oracle result, and proof gaps.
   After every verifier edit — not just support changes — refresh the
   task-local Dockerfile checksum label (e.g. `make harbor-sync`) and verify
   it matches `sha256sum` of the final `verifier.py`; a stale label fails
   `validate_task_topology` and blocks the task.

   If an Oracle run earns full mathematical correctness but zero evidence
   validity, diagnose the prose recognizer without reading or copying hidden
   answer text. Instrument or invoke the matcher to report only a boolean map
   of documented semantic clauses and contradiction checks. Do not print,
   tokenize, quote, or mine n-grams from the Oracle artifact. Repair the public
   semantic rule, write a fresh regression from the visible contract, and then
   rerun every check invalidated by the verifier change, including the exact
   Oracle.

Read [references/verifier-contract.md](references/verifier-contract.md) for the
detailed checklist and anti-pattern catalogue.

## Change hygiene

- Treat each task-local verifier-support copy as authoritative for that task.
  Do not silently synchronize a global helper into existing tasks. A shared
  support fix is an explicit selected-task migration; inspect the complete
  task-local diff, refresh only selected checksum labels, and run every
  affected Oracle.
- Treat support, schema, fixture, and adapter-digest changes as one dependency
  graph. A support change can alter task checksums and invalidate adapter locks;
  regenerate deterministic prospective artifacts and verify their scope.
- Give parallel agents disjoint task paths. The coordinating agent owns shared
  helpers, generated fan-out, full validation, staging, and integration.
- Stage explicit paths. Do not use `git add -A` after a broad sync or validation
  command, because generated drift and unrelated work can enter the commit.
- After each push, re-fetch review threads and compare comment IDs or creation
  revisions. An unchanged unresolved count does not prove that no new finding
  appeared; GitHub can re-anchor old comments to new lines.
- Verify PR state, head SHA, fork, and merge status before claiming a push or
  planning a fix. A commit pushed to a similarly named branch is not evidence
  that it entered a merged PR.

## Anti-patterns to reject

- Relying on JSON Schema alone while the verifier omits the same bounds or
  cardinalities.
- Using Python equality or `isinstance(x, int)` for schema-sensitive values
  when `bool` must be rejected.
- Converting unvalidated JSON numbers directly with `int()` or arithmetic;
  reject non-finite values and handle `ValueError`, `TypeError`, and
  `OverflowError` as ordinary invalid submissions.
- Sorting or constructing sets from nested values before validating that they
  are hashable and have the required scalar types.
- Reading an unbounded submission or hashing every evidence item before checking
  that the list has the required cardinality.
- Parsing or indexing the agent-visible input before proving byte-for-byte
  equality with the frozen verifier input.
- Replacing a hidden evidence marker with `bool(text.strip())` while evidence
  validity still contributes to reward.
- Treating a full Oracle reward as proof that malformed submissions are safe.
- Requiring undocumented lexical tokens or a preferred proof strategy.
- Treating prose context as independent regex hits; scope, target, and
  negation must remain associated across streamed chunks and alternate clause
  orderings.
- Declaring schema properties that the verifier does not enforce, such as a
  `max_bytes` evidence field that the resolver ignores. The visible schema,
  instructions, and executable checks must describe one contract.
- Returning one monolithic “correct” flag for all diagnostic dimensions.
- Mirroring the hidden solution in the fixture or testing private implementation
  text instead of observable verifier behavior.
- Resolving review threads because code “looks fixed” before the exact final
  tree and relevant checks provide evidence.
