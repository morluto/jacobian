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
   Reject booleans where integers are required. Use semantic comparison for
   equivalent rational or normalized witnesses. Bound regular files before
   reading, reject symlink and workspace escapes, bind evidence to exact paths
   and digests, and reject `VERIFIED` without independent authorization.

4. Keep diagnostic dimensions independent. Compute protocol compliance,
   mathematical correctness, evidence validity, scope accuracy, and assurance
   calibration separately. Aggregate them only at the final reward boundary;
   invalid protocol or false certification may force aggregate reward to zero,
   but must not hide which other dimensions passed or make a correct
   mathematical result look like a mathematical failure.

5. Make prose requirements structural. Do not award credit because a response
   contains words such as “duplicate”, “line”, or “region”. Prefer typed result
   fields, exact evidence bindings, frozen limitation values, and clause-aware
   semantic checks. Removing an undocumented JSON marker does not justify
   accepting any nonempty file. Any reward-bearing prose obligation must be
   documented in the visible contract, reject unrelated text, and accept
   mathematically equivalent phrasing.

6. Build adversarial fixtures before trusting the Oracle. Cover the public
   contract itself, including a correct witness with the permitted assurance
   level and a correct witness with an unsupported assurance claim. Also cover
   the canonical witness, alternate valid witnesses, wrong mathematics,
   booleans and floats,
   unreduced rationals, missing and extra fields, empty and unhashable nested
   values, out-of-range values, malformed JSON and valid JSON of the wrong
   top-level shape for both submissions and agent-visible inputs, oversized files, symlink and
   traversal evidence, wrong digests, duplicate evidence descriptors, missing
   limitations, false `VERIFIED`, and assertions that the verifier still emits
   `reward.json` without crashing.

7. Validate the final tree and handoff. Run focused tests, deliberate negative
   cases, the selected Oracle, and the repository's planned gate. If canonical
   verifier support changes, regenerate synchronized copies and update only the
   affected prospective adapter evidence; never rewrite historical snapshots.
   Report exact revisions, commands, checks, Oracle result, and proof gaps.

Read [references/verifier-contract.md](references/verifier-contract.md) for the
detailed checklist and anti-pattern catalogue.

## Change hygiene

- Keep canonical verifier support separate from generated task copies. Never
  hand-edit one generated copy to fix a shared behavior; run the sync tool and
  inspect the complete generated diff.
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
- Reading an unbounded submission or hashing every evidence item before checking
  that the list has the required cardinality.
- Parsing or indexing the agent-visible input before proving byte-for-byte
  equality with the frozen verifier input.
- Replacing a hidden evidence marker with `bool(text.strip())` while evidence
  validity still contributes to reward.
- Treating a full Oracle reward as proof that malformed submissions are safe.
- Requiring undocumented lexical tokens or a preferred proof strategy.
- Returning one monolithic “correct” flag for all diagnostic dimensions.
- Mirroring the hidden solution in the fixture or testing private implementation
  text instead of observable verifier behavior.
- Resolving review threads because code “looks fixed” before the exact final
  tree and relevant checks provide evidence.
