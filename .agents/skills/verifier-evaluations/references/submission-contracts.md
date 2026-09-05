# Submission contract details

Start from the mathematical predicate, then choose the smallest submission that
lets the verifier decide it:

- **Result only** when the verifier can replay the claim from frozen input.
- **Result plus witness** when a finite construction, counterexample, trace, or
  certificate is required for replay.
- **Formal proof** only when a supported checker accepts the language.

Put small structured mathematical certificates in `result`. Use a witness
artifact only for an external finite object that replay genuinely needs; it
must not duplicate the result or carry a natural-language explanation. Do not
add generic assurance claims, scope, completeness, limitations, or prose. If a
task exposes an independently authorized claim, publish the exact record that
authorizes it and reject a false claim. If a boundary affects score, use a
closed structured obligation ID, not a phrase, keyword count, or negation
heuristic.

Treat a typed result as the represented mathematical value, not a frozen JSON
layout or string rendering. Normalize and compare equivalent rational, scaled,
or unordered representations unless canonicalization is an explicit public
task outcome with an exact stated rule. `answer.txt` is never an authoritative
submission format.

Compute the scored predicate from the frozen input. Do not implement
correctness as equality with hidden `expected.json`. Do not compare a
normalized `Fraction` back to the submitted numerator and denominator. Do not
score formula strings or keyword-bearing prose when a task-local type or
already-derived enum exists. Generated family schemas must admit only the
certificate kinds that family can reward.

Independent result fields are independent predicates. A corrupted claimed
image does not decide whether two distinct points actually collide.

The visible schema and instructions must describe every field, type, bound,
scope rule, and witness requirement the verifier enforces. They must be
jointly satisfiable: an instruction-conforming object cannot be schema-valid
yet reward-ineligible because of an undocumented key set. Do not expose hidden
solutions, Oracle fixtures, or verifier implementation merely to explain the
format. A task may name the relevant Jacobian operations and their public
contracts as its experimental condition, but must not require a particular
discovery query, call order, or intermediate transcript unless that trace is
itself the mathematical object being checked. A hard task may deliberately
expose no operation that solves it today: that is a capability finding, not a
verifier defect.

The verifier derives conclusions from submitted mathematics. Do not score a
submitted copy of those conclusions, and do not publish them as schema
`const` fields. If a review asks to restore leaked constants so the schema and
verifier match, refuse and shrink the verifier and instruction instead. When
reducing a schema, update instruction, verifier key set, gold, public
contract, and host tests together. Closed `oneOf` success variants omit
inapplicable failure fields; they do not send JSON `null`. If one submitted
object implies a dimension, parse every related object at that derived size.
