# Lean semantic-guard scope assurance

This Assurance-family task freezes five cases from ATP Checkers
`test/PrefixScopeTests.lean` at commit
`3e7e99d027fece04d9cd96288cdd040c366458e5` (MIT).

Its single objective is assurance calibration: decide when a semantic-trap
finding is justified across full proof-state scope, pathological numeric
instances, proof-term exclusion, and independent zero-divisor versus truncation
semantics.  The task is **Hard (provisional)** because the correct labels
require a five-case semantic audit with type-sensitive and polarity-sensitive
distinctions; weaker agents are expected to rely on textual binder order or
literal `1`, while stronger agents reconstruct the checker boundary.

Shortcut audit: source labels are not included, case order carries no answer
pattern, and a single default warning policy fails several adversarial cases.
The verifier derives the expected findings from semantic facts and requires
case-specific reasons.  It reports `COMPUTED`; it does not run Lean or certify
the upstream linter implementation.
