# Propositional rewrite-trace replay

Reduce a frozen quantified-predicate instance to `False` by submitting a
sequence of independently replayable local AST rewrites.

This Regression benchmark is derived from `fol-traces/fol-traces`
`test/segment_0.jsonl` row 15 (`traindata_15`, rule `rule_1932381`) at immutable
revision `e0ae2fbfa163a29f8545db75b748555c2bc276e8` (MIT). Predicate applications
at the fixed variable are treated as propositional atoms, matching the source
trace boundary.

## Quality and shortcut audit

Quality score: **90/100**. The single objective is local rewrite-trace
reconstruction. Difficulty is **Hard (provisional)**: a valid trace requires
multiple nested rewrites, stable AST paths, associative normalization, and a
final contradiction; empirical calibration is pending. The verifier applies
each registered rule itself and accepts alternative valid rewrite orders, so a
copied source trace, direct `False`, or semantically equivalent shortcut cannot
pass.

Assurance is capped at `COMPUTED`: the verifier certifies the frozen
propositional AST and rule registry, not the source dataset's natural-language
predicate meanings or general first-order semantics.
