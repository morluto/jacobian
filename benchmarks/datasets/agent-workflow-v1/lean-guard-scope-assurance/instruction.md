# Audit Lean semantic-guard findings across scope and pathological types

For every frozen case in `/app/input.json`, determine the exact semantic-trap
findings that a sound checker should emit.  Preserve the input order and attach
the controlling reason for each decision.

This is an assurance-calibration task, not a request to run Lean or copy the
source repository's expected messages.  Your audit must distinguish full proof
state scope from textual binder order, ordinary numeric types from a custom
type where `0` and `1` are definitionally equal, proof terms from theorem
statements, and division-by-zero risk from precision-loss truncation.

Bind `/app/evidence/answer.txt`, include a `RESULT_JSON:` line containing the
submitted result as JSON, and bind that file with its SHA-256 digest.  The
offline verifier derives every expected finding from the frozen semantic facts.
Do not claim `VERIFIED` or claim that Lean compilation was replayed.
